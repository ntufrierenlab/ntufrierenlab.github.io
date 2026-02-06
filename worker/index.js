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

    let body;
    try {
      body = await request.json();
    } catch {
      return jsonResponse(400, { error: 'Invalid JSON' });
    }

    // Validate password
    if (!body.password || body.password !== env.AUTH_PASSWORD) {
      return jsonResponse(401, { error: 'Invalid password' });
    }

    // Route by action
    if (body.action === 'status') {
      return handleStatus(env);
    }

    return handleTrigger(body, env);
  },
};

async function handleStatus(env) {
  const repo = env.GITHUB_REPO || 'ntufrierenlab/ntufrierenlab.github.io';
  const apiUrl = `https://api.github.com/repos/${repo}/actions/workflows/add-paper.yml/runs?per_page=10`;

  const ghResponse = await fetch(apiUrl, {
    headers: {
      'Authorization': `Bearer ${env.GITHUB_PAT}`,
      'Accept': 'application/vnd.github.v3+json',
      'User-Agent': 'FrierenLab-Worker',
    },
  });

  if (!ghResponse.ok) {
    const detail = await ghResponse.text();
    return jsonResponse(ghResponse.status, { error: 'GitHub API error', detail });
  }

  const data = await ghResponse.json();
  const runs = (data.workflow_runs || []).map(run => ({
    id: run.id,
    status: run.status,
    conclusion: run.conclusion,
    created_at: run.created_at,
    updated_at: run.updated_at,
  }));

  return jsonResponse(200, { ok: true, runs });
}

async function handleTrigger(body, env) {
  const { arxiv_url, topic } = body;

  if (!arxiv_url) {
    return jsonResponse(400, { error: 'Missing arxiv_url' });
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
        arxiv_url: arxiv_url,
        topic: topic || 'General',
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
