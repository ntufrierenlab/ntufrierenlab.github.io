// Preprint Search + GitHub Actions Integration
// Uses OpenAlex API (free, no rate limit, CORS-friendly, covers arXiv & bioRxiv)
(function () {
  var searchInput = document.getElementById('arxiv-search-input');
  var searchBtn = document.getElementById('arxiv-search-btn');
  var resultsDiv = document.getElementById('arxiv-results');
  var loadingDiv = document.getElementById('arxiv-loading');
  var emptyDiv = document.getElementById('arxiv-empty');
  if (!searchInput) return;

  var API_BASE = 'https://api.openalex.org/works';
  var PAGE_SIZE = 10;
  var currentPapers = [];
  var currentPage = 1;
  var ARXIV_SOURCE = 's4306400194';
  var SELECT_FIELDS = 'id,title,authorships,publication_date,primary_location,locations,best_oa_location,doi,cited_by_count,abstract_inverted_index';
  var DEFAULT_TOPICS = ['Auto White Balance'];

  // ── Topic Management ──────────────────────────────────────────
  function getTopics() {
    var stored = localStorage.getItem('kb-topics');
    var topics;
    if (stored) {
      try { topics = JSON.parse(stored); } catch (e) { topics = DEFAULT_TOPICS.slice(); }
    } else {
      topics = DEFAULT_TOPICS.slice();
    }
    // Merge Hugo-rendered topics so localStorage stays in sync with actual taxonomy
    var changed = false;
    var hugoItems = document.querySelectorAll('#sidebar-topic-list li[data-hugo-topic]');
    hugoItems.forEach(function (li) {
      var name = li.getAttribute('data-hugo-topic');
      if (name && !topics.some(function (t) { return t.toLowerCase() === name.toLowerCase(); })) {
        topics.push(name);
        changed = true;
      }
    });
    if (changed) localStorage.setItem('kb-topics', JSON.stringify(topics));
    return topics;
  }

  function saveTopics(topics) {
    localStorage.setItem('kb-topics', JSON.stringify(topics));
  }

  function addTopic(name) {
    var trimmed = name.trim();
    if (!trimmed) return false;
    var topics = getTopics();
    // Check duplicate (case-insensitive)
    var exists = topics.some(function (t) { return t.toLowerCase() === trimmed.toLowerCase(); });
    if (exists) return false;
    topics.push(trimmed);
    saveTopics(topics);
    return true;
  }

  function removeTopic(name) {
    var topics = getTopics().filter(function (t) { return t !== name; });
    saveTopics(topics);
  }

  // ── Search ────────────────────────────────────────────────────
  searchBtn.addEventListener('click', doSearch);
  searchInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') doSearch();
  });

  function doSearch() {
    var query = searchInput.value.trim();
    if (query.length < 2) return;

    resultsDiv.innerHTML = '';
    removePagination();
    emptyDiv.style.display = 'none';
    loadingDiv.style.display = 'flex';
    currentPapers = [];
    currentPage = 1;

    var url = API_BASE +
      '?filter=title.search:' + encodeURIComponent(query) + ',is_oa:true' +
      '&sort=publication_date:desc' +
      '&select=' + SELECT_FIELDS +
      '&per_page=100' +
      '&mailto=ntufrierenlab@gmail.com';

    fetch(url)
    .then(function (r) { return r.ok ? r.json() : { results: [] }; })
    .then(function (data) {
      loadingDiv.style.display = 'none';

      var results = data.results || [];

      // Keep only papers where extractPaperInfo returns non-null
      // Preserve OpenAlex relevance order
      var seen = {};
      var merged = [];
      results.forEach(function (w) {
        if (!seen[w.id] && extractPaperInfo(w)) {
          seen[w.id] = true;
          merged.push(w);
        }
      });

      if (merged.length === 0) {
        emptyDiv.style.display = 'block';
        return;
      }
      currentPapers = merged;
      currentPage = 1;
      renderPage();
    })
    .catch(function (err) {
      loadingDiv.style.display = 'none';
      resultsDiv.innerHTML = '<p class="arxiv-error">Search failed: ' + err.message + '. Please try again.</p>';
    });
  }

  // ── Abstract reconstruction ───────────────────────────────────
  function reconstructAbstract(invertedIndex) {
    if (!invertedIndex) return '';
    var words = [];
    Object.keys(invertedIndex).forEach(function (word) {
      invertedIndex[word].forEach(function (pos) {
        words[pos] = word;
      });
    });
    var text = words.join(' ');
    if (text.length > 280) text = text.substring(0, 280) + '...';
    return text;
  }

  // ── Paper info extraction ─────────────────────────────────────
  function extractArxivFromUrls(allUrls) {
    var m1 = allUrls.match(/arxiv\.org\/(?:abs|pdf)\/([0-9]+\.[0-9]+)/);
    if (m1) return { source: 'arXiv', id: m1[1], absUrl: 'https://arxiv.org/abs/' + m1[1], pdfUrl: 'https://arxiv.org/pdf/' + m1[1] };
    var m2 = allUrls.match(/10\.48550\/arxiv\.([0-9]+\.[0-9]+)/i);
    if (m2) return { source: 'arXiv', id: m2[1], absUrl: 'https://arxiv.org/abs/' + m2[1], pdfUrl: 'https://arxiv.org/pdf/' + m2[1] };
    return null;
  }

  function identifySource(venueName) {
    if (!venueName) return 'Paper';
    var n = venueName.toLowerCase();
    var maps = [
      [/cvpr|computer vision and pattern recognition/, 'CVPR'],
      [/\biccv\b|international conference on computer vision(?! and pattern)/, 'ICCV'],
      [/\beccv\b|european conference on computer vision/, 'ECCV'],
      [/\bwacv\b|winter conference on applications of computer vision/, 'WACV'],
      [/\baccv\b|asian conference on computer vision/, 'ACCV'],
      [/neurips|neural information processing/, 'NeurIPS'],
      [/\bicml\b|international conference on machine learning/, 'ICML'],
      [/\biclr\b|international conference on learning representations/, 'ICLR'],
      [/\baaai\b/, 'AAAI'],
      [/siggraph asia/, 'SIGGRAPH Asia'],
      [/siggraph/, 'SIGGRAPH'],
      [/\bijcv\b|international journal of computer vision/, 'IJCV'],
      [/\btpami\b|transactions on pattern analysis/, 'TPAMI'],
    ];
    for (var i = 0; i < maps.length; i++) {
      if (maps[i][0].test(n)) return maps[i][1];
    }
    return 'Paper';
  }

  function extractPaperInfo(work) {
    var loc = work.primary_location;
    var locations = work.locations || [];

    // 1) Check primary_location for arXiv
    if (loc) {
      var pageUrl = (loc.landing_page_url || '');
      var locId = (loc.id || '');
      var pdfField = (loc.pdf_url || '');
      var allUrls = pageUrl + ' ' + locId + ' ' + pdfField;

      var arxiv = extractArxivFromUrls(allUrls);
      if (arxiv) return arxiv;

      // 2) Check primary_location for bioRxiv
      if (/biorxiv\.org/i.test(allUrls)) {
        var b1 = allUrls.match(/biorxiv\.org\/content\/(?:biorxiv\/early\/[0-9/]+\/)?([0-9]+\.[0-9]+\/[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]+)(v[0-9]+)?/);
        if (b1) {
          var bdoi = b1[1];
          var ver = b1[2] || 'v1';
          return { source: 'bioRxiv', id: bdoi, absUrl: 'https://www.biorxiv.org/content/' + bdoi + ver, pdfUrl: 'https://www.biorxiv.org/content/' + bdoi + ver + '.full.pdf' };
        }
        if (/biorxiv\.org/.test(pdfField)) {
          var absGuess = pdfField.replace(/\.full\.pdf$/, '');
          var doiMatch = pdfField.match(/([0-9]+\.[0-9]+\/[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]+)/);
          var displayId = doiMatch ? doiMatch[1] : 'preprint';
          return { source: 'bioRxiv', id: displayId, absUrl: absGuess, pdfUrl: pdfField };
        }
      }

      // 3) Check if primary_location is a recognized top conference/journal
      var venueName = (loc.source && loc.source.display_name) ? loc.source.display_name : '';
      var sourceLabel = identifySource(venueName);
      if (sourceLabel !== 'Paper') {
        // It's a recognized venue — find arXiv URL from locations if available
        var arxivInfo = null;
        for (var i = 0; i < locations.length; i++) {
          var l = locations[i];
          var lUrls = (l.landing_page_url || '') + ' ' + (l.id || '') + ' ' + (l.pdf_url || '');
          arxivInfo = extractArxivFromUrls(lUrls);
          if (arxivInfo) break;
        }

        // Find PDF URL
        var confPdf = '';
        var boa = work.best_oa_location;
        if (boa && boa.pdf_url) confPdf = boa.pdf_url;
        if (!confPdf) {
          for (var j = 0; j < locations.length; j++) {
            if (locations[j].pdf_url) { confPdf = locations[j].pdf_url; break; }
          }
        }
        if (!confPdf && arxivInfo) confPdf = arxivInfo.pdfUrl;
        if (!confPdf) return null;

        var doi = work.doi || '';
        var cleanDoi = doi.replace(/^https?:\/\/doi\.org\//, '');
        // Prefer arXiv URL for the link (existing pipeline), otherwise landing page
        var absUrl = arxivInfo ? arxivInfo.absUrl : (loc.landing_page_url || 'https://doi.org/' + cleanDoi);
        var displayId = arxivInfo ? arxivInfo.id : cleanDoi;

        return {
          source: sourceLabel,
          id: displayId,
          absUrl: absUrl,
          pdfUrl: confPdf,
          doi: cleanDoi,
          arxivUrl: arxivInfo ? arxivInfo.absUrl : '',
          isConference: true
        };
      }
    }

    // 4) Check all locations for an arXiv version (non-conference papers with arXiv copy)
    for (var k = 0; k < locations.length; k++) {
      var ll = locations[k];
      var llUrls = (ll.landing_page_url || '') + ' ' + (ll.id || '') + ' ' + (ll.pdf_url || '');
      var arxivFromLoc = extractArxivFromUrls(llUrls);
      if (arxivFromLoc) return arxivFromLoc;
    }

    return null;
  }

  // ── Render page ───────────────────────────────────────────────
  function renderPage() {
    resultsDiv.innerHTML = '';
    var totalPages = Math.ceil(currentPapers.length / PAGE_SIZE);
    var start = (currentPage - 1) * PAGE_SIZE;
    var end = Math.min(start + PAGE_SIZE, currentPapers.length);
    var pagePapers = currentPapers.slice(start, end);

    pagePapers.forEach(function (work) {
      var info = extractPaperInfo(work);
      var authors = (work.authorships || []).map(function (a) {
        return a.author ? a.author.display_name : '';
      }).filter(Boolean).join(', ');

      var pubDate = work.publication_date || '';
      var citations = work.cited_by_count || 0;
      var abstract = reconstructAbstract(work.abstract_inverted_index);
      var title = work.title || 'Untitled';
      var badgeClass = info.source === 'bioRxiv' ? ' badge-biorxiv' : (info.isConference ? ' badge-conf' : '');
      var displayId = info.id;
      // Truncate long DOIs for display
      if (displayId.length > 30) displayId = displayId.substring(0, 30) + '...';

      // Build add-button data attributes
      var addAttrs = 'data-url="' + escapeAttr(info.absUrl) + '" data-title="' + escapeAttr(title) + '"';
      if (info.doi) addAttrs += ' data-doi="' + escapeAttr(info.doi) + '"';
      if (info.isConference) addAttrs += ' data-pdf="' + escapeAttr(info.pdfUrl) + '"';
      if (info.arxivUrl) addAttrs += ' data-arxiv="' + escapeAttr(info.arxivUrl) + '"';

      var card = document.createElement('div');
      card.className = 'arxiv-result-card';
      card.innerHTML =
        '<div class="arxiv-result-header">' +
          '<div class="arxiv-result-meta">' +
            (pubDate ? '<span class="arxiv-result-date">' + pubDate + '</span>' : '') +
            '<span class="arxiv-result-citations">' + citations + ' citations</span>' +
            '<span class="arxiv-result-id' + badgeClass + '">' + info.source + ': ' + escapeHtml(displayId) + '</span>' +
          '</div>' +
        '</div>' +
        '<h3 class="arxiv-result-title">' + escapeHtml(title) + '</h3>' +
        '<p class="arxiv-result-authors">' + escapeHtml(authors) + '</p>' +
        (abstract ? '<p class="arxiv-result-abstract">' + escapeHtml(abstract) + '</p>' : '') +
        '<div class="arxiv-result-actions">' +
          '<a href="' + info.absUrl + '" class="btn btn-outline btn-sm" target="_blank" rel="noopener">' +
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>' +
            info.source + '</a>' +
          '<button class="btn btn-primary btn-sm add-paper-btn"' +
            (sessionStorage.getItem('kb-session-pwd') ? '' : ' style="display:none"') +
            ' ' + addAttrs + '>' +
            '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>' +
            'Add to Knowledge Base</button>' +
        '</div>';

      resultsDiv.appendChild(card);
    });

    // Attach add-paper handlers — now opens topic picker first
    var addBtns = resultsDiv.querySelectorAll('.add-paper-btn');
    addBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var url = this.getAttribute('data-url');
        var title = this.getAttribute('data-title');
        var doi = this.getAttribute('data-doi') || '';
        var pdf = this.getAttribute('data-pdf') || '';
        var arxivUrl = this.getAttribute('data-arxiv') || '';
        openTopicPicker(url, title, this, doi, pdf, arxivUrl);
      });
    });

    renderPagination(totalPages);
  }

  // ── Topic Picker Modal ────────────────────────────────────────
  var topicModal = document.getElementById('topic-modal');
  var topicListEl = document.getElementById('topic-list');
  var newTopicInput = document.getElementById('new-topic-input');
  var newTopicAddBtn = document.getElementById('new-topic-add');
  var topicCancelBtn = document.getElementById('topic-cancel');
  var topicConfirmBtn = document.getElementById('topic-confirm');
  var topicOverlay = topicModal ? topicModal.querySelector('.modal-overlay') : null;

  var pendingPaper = { url: '', title: '', btn: null, doi: '', pdf: '', arxivUrl: '' };

  function openTopicPicker(url, title, btn, doi, pdf, arxivUrl) {
    if (!sessionStorage.getItem('kb-session-pwd')) {
      alert('Please enter the password first (click the lock icon in the sidebar).');
      return;
    }
    pendingPaper = { url: url, title: title, btn: btn, doi: doi || '', pdf: pdf || '', arxivUrl: arxivUrl || '' };
    renderTopicList();
    newTopicInput.value = '';
    topicModal.style.display = 'flex';
  }

  function renderTopicList(selectedTopic) {
    var topics = getTopics();
    topicListEl.innerHTML = '';
    topics.forEach(function (t) {
      var label = document.createElement('label');
      label.className = 'topic-picker-item';
      var isSelected = selectedTopic ? (t === selectedTopic) : false;
      label.innerHTML =
        '<input type="radio" name="topic-pick" value="' + escapeAttr(t) + '"' + (isSelected ? ' checked' : '') + '>' +
        '<span class="topic-picker-radio"></span>' +
        '<span class="topic-picker-name">' + escapeHtml(t) + '</span>' +
        '<button class="topic-picker-delete" data-topic="' + escapeAttr(t) + '" title="Remove topic">' +
          '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>' +
        '</button>';
      topicListEl.appendChild(label);
    });

    // If no topic was pre-selected, select the first one
    if (!selectedTopic) {
      var firstRadio = topicListEl.querySelector('input[type="radio"]');
      if (firstRadio) firstRadio.checked = true;
    }

    // Attach delete handlers
    var delBtns = topicListEl.querySelectorAll('.topic-picker-delete');
    delBtns.forEach(function (db) {
      db.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        var topicName = this.getAttribute('data-topic');
        if (getTopics().length <= 1) return; // keep at least one
        removeTopic(topicName);
        renderTopicList();
        refreshSidebarTopics();
      });
    });
  }

  if (newTopicAddBtn) {
    newTopicAddBtn.addEventListener('click', function () {
      var name = newTopicInput.value.trim();
      if (name && addTopic(name)) {
        newTopicInput.value = '';
        renderTopicList(name);
        refreshSidebarTopics();
      }
    });
  }

  if (newTopicInput) {
    newTopicInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        newTopicAddBtn.click();
      }
    });
  }

  function closeTopicModal() {
    topicModal.style.display = 'none';
  }

  if (topicCancelBtn) topicCancelBtn.addEventListener('click', closeTopicModal);
  if (topicOverlay) topicOverlay.addEventListener('click', closeTopicModal);

  if (topicConfirmBtn) {
    topicConfirmBtn.addEventListener('click', function () {
      var selected = topicListEl.querySelector('input[type="radio"]:checked');
      var topic = selected ? selected.value : 'General';
      closeTopicModal();
      addPaper(pendingPaper.url, pendingPaper.title, pendingPaper.btn, topic, pendingPaper.doi, pendingPaper.pdf, pendingPaper.arxivUrl);
    });
  }

  // ── Sidebar topic sync ────────────────────────────────────────
  function refreshSidebarTopics() {
    var list = document.getElementById('sidebar-topic-list');
    if (!list) return;

    // Collect Hugo-rendered topics (have paper counts)
    var hugoTopics = {};
    list.querySelectorAll('li[data-hugo-topic]').forEach(function (li) {
      hugoTopics[li.getAttribute('data-hugo-topic')] = li;
    });

    // Remove old JS-injected items
    list.querySelectorAll('li[data-js-topic]').forEach(function (li) {
      li.remove();
    });

    // Add user topics that aren't already shown by Hugo
    var topics = getTopics();
    topics.forEach(function (t) {
      if (!hugoTopics[t]) {
        var li = document.createElement('li');
        li.setAttribute('data-js-topic', t);
        li.innerHTML =
          '<a href="javascript:void(0)">' +
            '<span class="topic-dot"></span>' +
            escapeHtml(t) +
            '<span class="topic-count">0</span>' +
          '</a>';
        list.appendChild(li);
      }
    });
  }

  // Sidebar add-topic form
  var sidebarAddBtn = document.getElementById('sidebar-add-topic');
  var sidebarTopicForm = document.getElementById('sidebar-topic-form');
  var sidebarTopicInput = document.getElementById('sidebar-topic-input');
  var sidebarTopicSave = document.getElementById('sidebar-topic-save');
  var sidebarTopicCancel = document.getElementById('sidebar-topic-cancel');

  if (sidebarAddBtn) {
    sidebarAddBtn.addEventListener('click', function () {
      sidebarTopicForm.style.display = 'block';
      sidebarTopicInput.value = '';
      sidebarTopicInput.focus();
    });
  }
  if (sidebarTopicCancel) {
    sidebarTopicCancel.addEventListener('click', function () {
      sidebarTopicForm.style.display = 'none';
    });
  }
  if (sidebarTopicSave) {
    sidebarTopicSave.addEventListener('click', function () {
      var name = sidebarTopicInput.value.trim();
      if (name && addTopic(name)) {
        sidebarTopicForm.style.display = 'none';
        refreshSidebarTopics();
      }
    });
  }
  if (sidebarTopicInput) {
    sidebarTopicInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        sidebarTopicSave.click();
      }
    });
  }

  // Init sidebar topics on load
  refreshSidebarTopics();

  // ── Pagination ────────────────────────────────────────────────
  function renderPagination(totalPages) {
    removePagination();
    if (totalPages <= 1) return;

    var nav = document.createElement('div');
    nav.className = 'pagination';
    nav.id = 'arxiv-pagination';

    var prevBtn = document.createElement('button');
    prevBtn.className = 'pagination-btn';
    prevBtn.textContent = 'Previous';
    prevBtn.disabled = currentPage === 1;
    prevBtn.addEventListener('click', function () {
      if (currentPage > 1) { currentPage--; renderPage(); scrollToResults(); }
    });
    nav.appendChild(prevBtn);

    var pages = getPageNumbers(currentPage, totalPages);
    pages.forEach(function (p) {
      if (p === '...') {
        var dots = document.createElement('span');
        dots.className = 'pagination-dots';
        dots.textContent = '...';
        nav.appendChild(dots);
      } else {
        var pageBtn = document.createElement('button');
        pageBtn.className = 'pagination-btn' + (p === currentPage ? ' active' : '');
        pageBtn.textContent = p;
        pageBtn.addEventListener('click', function () {
          currentPage = p;
          renderPage();
          scrollToResults();
        });
        nav.appendChild(pageBtn);
      }
    });

    var nextBtn = document.createElement('button');
    nextBtn.className = 'pagination-btn';
    nextBtn.textContent = 'Next';
    nextBtn.disabled = currentPage === totalPages;
    nextBtn.addEventListener('click', function () {
      if (currentPage < totalPages) { currentPage++; renderPage(); scrollToResults(); }
    });
    nav.appendChild(nextBtn);

    var info = document.createElement('div');
    info.className = 'pagination-info';
    info.textContent = currentPapers.length + ' papers found';
    nav.appendChild(info);

    resultsDiv.parentNode.insertBefore(nav, resultsDiv.nextSibling);
  }

  function getPageNumbers(current, total) {
    if (total <= 7) {
      var arr = [];
      for (var i = 1; i <= total; i++) arr.push(i);
      return arr;
    }
    var pages = [1];
    if (current > 3) pages.push('...');
    for (var j = Math.max(2, current - 1); j <= Math.min(total - 1, current + 1); j++) {
      pages.push(j);
    }
    if (current < total - 2) pages.push('...');
    pages.push(total);
    return pages;
  }

  function removePagination() {
    var old = document.getElementById('arxiv-pagination');
    if (old) old.remove();
  }

  function scrollToResults() {
    var header = document.querySelector('.search-page-header');
    if (header) header.scrollIntoView({ behavior: 'smooth' });
  }

  // ── Credentials ─────────────────────────────────────────────────
  // Clean up any old localStorage entries from previous versions
  localStorage.removeItem('kb-password');
  localStorage.removeItem('worker-url');

  var WORKER_URL = 'https://frieren-lab-proxy.ntufrierenlab.workers.dev';

  // ── Add paper (via Cloudflare Worker proxy) ────────────────────
  function addPaper(paperUrl, title, btn, topic, doi, pdfUrl, arxivUrl) {
    btn.disabled = true;
    btn.innerHTML =
      '<div class="spinner-small"></div>' +
      'Processing...';

    var password = sessionStorage.getItem('kb-session-pwd') || '';
    var workerUrl = WORKER_URL;

    // For conference papers with arXiv version, use arXiv URL for the pipeline
    var payload = {
      password: password,
      arxiv_url: arxivUrl || paperUrl,
      topic: topic
    };
    // Only send doi/pdf_url for conference papers WITHOUT arXiv
    if (doi && !arxivUrl) {
      payload.doi = doi;
      if (pdfUrl) payload.pdf_url = pdfUrl;
    }

    fetch(workerUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data.ok) {
        btn.innerHTML =
          '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>' +
          'Queued!';
        btn.classList.add('btn-success');
        showToast();

        // Save pending paper for notification tracking
        var pending = [];
        try { pending = JSON.parse(sessionStorage.getItem('kb-pending-papers') || '[]'); } catch (e) { pending = []; }
        pending.push({
          title: title,
          arxivUrl: paperUrl,
          topic: topic,
          triggeredAt: new Date().toISOString(),
          status: 'pending',
          runId: null,
          readByUser: false
        });
        sessionStorage.setItem('kb-pending-papers', JSON.stringify(pending));
        window.dispatchEvent(new CustomEvent('kb-paper-added'));
      } else {
        throw new Error(data.error || 'Failed to trigger workflow');
      }
    })
    .catch(function (err) {
      btn.disabled = false;
      btn.innerHTML =
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>' +
        'Add to Knowledge Base';
      alert('Error: ' + err.message + '\n\nPlease check your password and settings.');
    });
  }

  function showToast() {
    var toast = document.getElementById('process-toast');
    toast.style.display = 'flex';
    setTimeout(function () {
      toast.style.display = 'none';
    }, 8000);
  }

  // ── Helpers ───────────────────────────────────────────────────
  function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function escapeAttr(s) {
    return s.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // Show/hide add buttons when auth state changes
  window.addEventListener('kb-auth-changed', function (e) {
    var btns = resultsDiv.querySelectorAll('.add-paper-btn');
    var show = e.detail.authenticated;
    btns.forEach(function (btn) {
      if (!btn.classList.contains('btn-success')) {
        btn.style.display = show ? '' : 'none';
      }
    });
  });

})();
