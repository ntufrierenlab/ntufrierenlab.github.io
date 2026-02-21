// Cloudflare Worker — proxy for triggering GitHub Actions with password auth.
// Secrets (set via `wrangler secret put`):
//   AUTH_PASSWORD  — simple password users enter on the website
//   GITHUB_PAT    — GitHub PAT with repo scope
//   GITHUB_REPO   — owner/repo, e.g. "ntufrierenlab/ntufrierenlab.github.io"

export default {
  async fetch(request, env) {
    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    if (request.method !== 'POST') {
      return jsonResponse(405, { error: 'Method not allowed' });
    }

    // Route multipart/form-data uploads before JSON parsing
    const contentType = request.headers.get('content-type') || '';
    if (contentType.includes('multipart/form-data')) {
      return handleUpload(request, env);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return jsonResponse(400, { error: 'Invalid JSON' });
    }

    // Public actions (no password required)
    if (body.action === 'search-arxiv') {
      return handleSearchArxiv(body);
    }

    // Validate password
    if (!body.password || body.password !== env.AUTH_PASSWORD) {
      return jsonResponse(401, { error: 'Invalid password' });
    }

    // Route by action
    if (body.action === 'status') {
      return handleStatus(env);
    }

    if (body.action === 'delete') {
      return handleDelete(body, env);
    }

    if (body.action === 'add-note') {
      return handleAddNote(body, env);
    }

    if (body.action === 'delete-note') {
      return handleDeleteNote(body, env);
    }

    if (body.action === 'update-topics') {
      return handleUpdateTopics(body, env);
    }

    return handleTrigger(body, env);
  },
};

// Simple in-memory rate limiter for uploads: max 5 per hour per IP
const uploadRateMap = new Map();
const UPLOAD_RATE_LIMIT = 5;
const UPLOAD_RATE_WINDOW = 60 * 60 * 1000; // 1 hour

function checkUploadRate(ip) {
  const now = Date.now();
  const entry = uploadRateMap.get(ip);
  if (!entry) {
    uploadRateMap.set(ip, { count: 1, resetAt: now + UPLOAD_RATE_WINDOW });
    return true;
  }
  if (now > entry.resetAt) {
    uploadRateMap.set(ip, { count: 1, resetAt: now + UPLOAD_RATE_WINDOW });
    return true;
  }
  if (entry.count >= UPLOAD_RATE_LIMIT) return false;
  entry.count++;
  return true;
}

async function handleUpload(request, env) {
  // Rate limit uploads
  const clientIP = request.headers.get('cf-connecting-ip') || 'unknown';
  if (!checkUploadRate(clientIP)) {
    return jsonResponse(429, { error: 'Too many uploads. Please try again later.' });
  }

  let formData;
  try {
    formData = await request.formData();
  } catch {
    return jsonResponse(400, { error: 'Invalid form data' });
  }

  const password = formData.get('password');
  if (!password || password !== env.AUTH_PASSWORD) {
    return jsonResponse(401, { error: 'Invalid password' });
  }

  const file = formData.get('file');
  const title = (formData.get('title') || '').trim();
  const topic = formData.get('topic') || 'General';

  if (!file || typeof file === 'string') {
    return jsonResponse(400, { error: 'Missing PDF file' });
  }
  if (!title) {
    return jsonResponse(400, { error: 'Missing paper title' });
  }
  if (file.type !== 'application/pdf') {
    return jsonResponse(400, { error: 'Only PDF files are accepted' });
  }
  const MAX_SIZE = 50 * 1024 * 1024; // 50MB
  if (file.size > MAX_SIZE) {
    return jsonResponse(400, { error: 'File too large (max 50MB)' });
  }

  // Store in R2
  const key = crypto.randomUUID() + '.pdf';
  await env.PDF_BUCKET.put(key, file.stream(), {
    httpMetadata: { contentType: 'application/pdf' },
    customMetadata: { title: title, uploadedAt: new Date().toISOString() },
  });

  const publicUrl = (env.R2_PUBLIC_URL || '').replace(/\/$/, '') + '/' + key;

  // Trigger add-paper.yml with upload inputs
  const repo = env.GITHUB_REPO || 'ntufrierenlab/ntufrierenlab.github.io';
  const apiUrl = `https://api.github.com/repos/${repo}/actions/workflows/add-paper.yml/dispatches`;

  const ghResponse = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.GITHUB_PAT}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'User-Agent': 'FrierenLab-Worker',
    },
    body: JSON.stringify({
      ref: 'main',
      inputs: {
        arxiv_url: '',
        topic: topic,
        doi: '',
        pdf_url: publicUrl,
        source_type: 'upload',
        paper_title: title,
      },
    }),
  });

  if (ghResponse.status === 204) {
    return jsonResponse(200, { ok: true, message: 'Upload received, workflow triggered' });
  }

  const ghData = await ghResponse.text();
  return jsonResponse(ghResponse.status, { error: 'GitHub API error', detail: ghData });
}

async function handleStatus(env) {
  const repo = env.GITHUB_REPO || 'ntufrierenlab/ntufrierenlab.github.io';
  const ghHeaders = {
    'Authorization': `Bearer ${env.GITHUB_PAT}`,
    'Accept': 'application/vnd.github.v3+json',
    'User-Agent': 'FrierenLab-Worker',
  };

  // Query all workflow runs in parallel
  const [addRes, deleteRes, noteRes, topicsRes, deployRes] = await Promise.all([
    fetch(`https://api.github.com/repos/${repo}/actions/workflows/add-paper.yml/runs?per_page=10`, { headers: ghHeaders }),
    fetch(`https://api.github.com/repos/${repo}/actions/workflows/delete-paper.yml/runs?per_page=10`, { headers: ghHeaders }),
    fetch(`https://api.github.com/repos/${repo}/actions/workflows/add-note.yml/runs?per_page=10`, { headers: ghHeaders }),
    fetch(`https://api.github.com/repos/${repo}/actions/workflows/update-topics.yml/runs?per_page=10`, { headers: ghHeaders }),
    fetch(`https://api.github.com/repos/${repo}/actions/workflows/deploy.yml/runs?per_page=5`, { headers: ghHeaders }),
  ]);

  if (!addRes.ok) {
    const detail = await addRes.text();
    return jsonResponse(addRes.status, { error: 'GitHub API error', detail });
  }

  const addData = await addRes.json();
  const runs = (addData.workflow_runs || []).map(run => ({
    id: run.id,
    status: run.status,
    conclusion: run.conclusion,
    created_at: run.created_at,
    updated_at: run.updated_at,
  }));

  // Merge delete runs if available (workflow may not exist yet)
  if (deleteRes.ok) {
    const deleteData = await deleteRes.json();
    (deleteData.workflow_runs || []).forEach(run => {
      runs.push({
        id: run.id,
        status: run.status,
        conclusion: run.conclusion,
        created_at: run.created_at,
        updated_at: run.updated_at,
      });
    });
  }

  // Merge add-note runs if available
  if (noteRes.ok) {
    const noteData = await noteRes.json();
    (noteData.workflow_runs || []).forEach(run => {
      runs.push({
        id: run.id,
        status: run.status,
        conclusion: run.conclusion,
        created_at: run.created_at,
        updated_at: run.updated_at,
      });
    });
  }

  // Merge update-topics runs if available
  if (topicsRes.ok) {
    const topicsData = await topicsRes.json();
    (topicsData.workflow_runs || []).forEach(run => {
      runs.push({
        id: run.id,
        status: run.status,
        conclusion: run.conclusion,
        created_at: run.created_at,
        updated_at: run.updated_at,
      });
    });
  }

  // Collect deploy runs separately
  const deployRuns = [];
  if (deployRes.ok) {
    const deployData = await deployRes.json();
    (deployData.workflow_runs || []).forEach(run => {
      deployRuns.push({
        id: run.id,
        status: run.status,
        conclusion: run.conclusion,
        created_at: run.created_at,
        updated_at: run.updated_at,
      });
    });
  }

  return jsonResponse(200, { ok: true, runs, deployRuns });
}

async function handleDelete(body, env) {
  const { paper_filename } = body;

  if (!paper_filename) {
    return jsonResponse(400, { error: 'Missing paper_filename' });
  }

  // Validate filename: no path traversal
  if (paper_filename.includes('/') || paper_filename.includes('..')) {
    return jsonResponse(400, { error: 'Invalid filename' });
  }

  const repo = env.GITHUB_REPO || 'ntufrierenlab/ntufrierenlab.github.io';
  const apiUrl = `https://api.github.com/repos/${repo}/actions/workflows/delete-paper.yml/dispatches`;

  const ghResponse = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.GITHUB_PAT}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'User-Agent': 'FrierenLab-Worker',
    },
    body: JSON.stringify({
      ref: 'main',
      inputs: {
        paper_filename: paper_filename,
      },
    }),
  });

  if (ghResponse.status === 204) {
    return jsonResponse(200, { ok: true, message: 'Delete workflow triggered' });
  }

  const ghData = await ghResponse.text();
  return jsonResponse(ghResponse.status, {
    error: 'GitHub API error',
    detail: ghData,
  });
}

async function handleAddNote(body, env) {
  const { paper_filename, note_text } = body;

  if (!paper_filename) {
    return jsonResponse(400, { error: 'Missing paper_filename' });
  }
  if (!note_text || !note_text.trim()) {
    return jsonResponse(400, { error: 'Missing note_text' });
  }
  if (note_text.length > 2000) {
    return jsonResponse(400, { error: 'Note text too long (max 2000 chars)' });
  }
  if (paper_filename.includes('/') || paper_filename.includes('..')) {
    return jsonResponse(400, { error: 'Invalid filename' });
  }

  const repo = env.GITHUB_REPO || 'ntufrierenlab/ntufrierenlab.github.io';
  const apiUrl = `https://api.github.com/repos/${repo}/actions/workflows/add-note.yml/dispatches`;

  const ghResponse = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.GITHUB_PAT}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'User-Agent': 'FrierenLab-Worker',
    },
    body: JSON.stringify({
      ref: 'main',
      inputs: {
        paper_filename: paper_filename,
        note_text: note_text.trim(),
      },
    }),
  });

  if (ghResponse.status === 204) {
    return jsonResponse(200, { ok: true, message: 'Note workflow triggered' });
  }

  const ghData = await ghResponse.text();
  return jsonResponse(ghResponse.status, {
    error: 'GitHub API error',
    detail: ghData,
  });
}

async function handleDeleteNote(body, env) {
  const { paper_filename, note_date } = body;

  if (!paper_filename) {
    return jsonResponse(400, { error: 'Missing paper_filename' });
  }
  if (!note_date) {
    return jsonResponse(400, { error: 'Missing note_date' });
  }
  if (paper_filename.includes('/') || paper_filename.includes('..')) {
    return jsonResponse(400, { error: 'Invalid filename' });
  }

  const repo = env.GITHUB_REPO || 'ntufrierenlab/ntufrierenlab.github.io';
  const apiUrl = `https://api.github.com/repos/${repo}/actions/workflows/add-note.yml/dispatches`;

  const ghResponse = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.GITHUB_PAT}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'User-Agent': 'FrierenLab-Worker',
    },
    body: JSON.stringify({
      ref: 'main',
      inputs: {
        paper_filename: paper_filename,
        note_text: '__DELETE__:' + note_date,
      },
    }),
  });

  if (ghResponse.status === 204) {
    return jsonResponse(200, { ok: true, message: 'Delete note workflow triggered' });
  }

  const ghData = await ghResponse.text();
  return jsonResponse(ghResponse.status, {
    error: 'GitHub API error',
    detail: ghData,
  });
}

async function handleUpdateTopics(body, env) {
  const { paper_filename, topics } = body;

  if (!paper_filename) {
    return jsonResponse(400, { error: 'Missing paper_filename' });
  }
  if (!topics) {
    return jsonResponse(400, { error: 'Missing topics' });
  }
  if (paper_filename.includes('/') || paper_filename.includes('..')) {
    return jsonResponse(400, { error: 'Invalid filename' });
  }

  const repo = env.GITHUB_REPO || 'ntufrierenlab/ntufrierenlab.github.io';
  const apiUrl = `https://api.github.com/repos/${repo}/actions/workflows/update-topics.yml/dispatches`;

  const ghResponse = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.GITHUB_PAT}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'User-Agent': 'FrierenLab-Worker',
    },
    body: JSON.stringify({
      ref: 'main',
      inputs: {
        paper_filename: paper_filename,
        topics: topics,
      },
    }),
  });

  if (ghResponse.status === 204) {
    return jsonResponse(200, { ok: true, message: 'Topics update triggered' });
  }

  const ghData = await ghResponse.text();
  return jsonResponse(ghResponse.status, {
    error: 'GitHub API error',
    detail: ghData,
  });
}

async function handleSearchArxiv(body) {
  const { query } = body;
  if (!query || query.length < 2) {
    return jsonResponse(400, { error: 'Missing or too short query' });
  }

  const arxivUrl = 'https://export.arxiv.org/api/query' +
    '?search_query=ti:' + encodeURIComponent(query) +
    '&max_results=20&sortBy=submittedDate&sortOrder=descending';

  // Retry once on 429 (arXiv rate limit is 3s between requests)
  let res = await fetch(arxivUrl, {
    headers: { 'User-Agent': 'FrierenLab-Worker' },
  });

  if (res.status === 429) {
    await new Promise(r => setTimeout(r, 4000));
    res = await fetch(arxivUrl, {
      headers: { 'User-Agent': 'FrierenLab-Worker' },
    });
  }

  if (!res.ok) {
    return jsonResponse(res.status, { error: 'arXiv API error' });
  }

  const xml = await res.text();
  return new Response(xml, {
    status: 200,
    headers: {
      'Content-Type': 'application/xml',
      ...corsHeaders(),
    },
  });
}

async function handleTrigger(body, env) {
  const { arxiv_url, topic, doi, pdf_url } = body;

  if (!arxiv_url && !doi) {
    return jsonResponse(400, { error: 'Missing arxiv_url or doi' });
  }

  const repo = env.GITHUB_REPO || 'ntufrierenlab/ntufrierenlab.github.io';
  const apiUrl = `https://api.github.com/repos/${repo}/actions/workflows/add-paper.yml/dispatches`;

  const ghResponse = await fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.GITHUB_PAT}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'User-Agent': 'FrierenLab-Worker',
    },
    body: JSON.stringify({
      ref: 'main',
      inputs: {
        arxiv_url: arxiv_url || '',
        topic: topic || 'General',
        doi: doi || '',
        pdf_url: pdf_url || '',
        source_type: 'search',
        paper_title: '',
      },
    }),
  });

  if (ghResponse.status === 204) {
    return jsonResponse(200, { ok: true, message: 'Workflow triggered' });
  }

  const ghData = await ghResponse.text();
  return jsonResponse(ghResponse.status, {
    error: 'GitHub API error',
    detail: ghData,
  });
}

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
  };
}

function jsonResponse(status, data) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders(),
    },
  });
}
