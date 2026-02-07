// Wrap tables in scrollable containers
(function () {
  var tables = document.querySelectorAll('.paper-content table');
  for (var i = 0; i < tables.length; i++) {
    var wrapper = document.createElement('div');
    wrapper.className = 'table-scroll';
    tables[i].parentNode.insertBefore(wrapper, tables[i]);
    wrapper.appendChild(tables[i]);
  }
})();

// Theme toggle
(function () {
  var toggle = document.getElementById('theme-toggle');
  var html = document.documentElement;
  var stored = localStorage.getItem('theme');
  if (stored) {
    html.setAttribute('data-theme', stored);
  } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    html.setAttribute('data-theme', 'dark');
  }
  if (toggle) {
    toggle.addEventListener('click', function () {
      var current = html.getAttribute('data-theme');
      var next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
    });
  }
})();

// Sidebar auth button + password modal
(function () {
  var authBtn = document.getElementById('sidebar-auth-btn');
  var modal = document.getElementById('password-modal');
  if (!authBtn || !modal) return;

  var overlay = modal.querySelector('.modal-overlay');
  var passwordInput = document.getElementById('kb-password');
  var saveBtn = document.getElementById('password-save');
  var cancelBtn = document.getElementById('password-cancel');

  // Reflect current auth state
  function updateAuthState() {
    var authed = !!sessionStorage.getItem('kb-session-pwd');
    if (authed) {
      authBtn.classList.add('authenticated');
      authBtn.setAttribute('aria-label', 'Authenticated');
    } else {
      authBtn.classList.remove('authenticated');
      authBtn.setAttribute('aria-label', 'Authentication');
    }
    // Show/hide delete button on paper pages
    var deleteBtn = document.getElementById('btn-delete-paper');
    if (deleteBtn) {
      deleteBtn.style.display = authed ? '' : 'none';
    }
    // Show/hide sidebar manage-topics button
    var manageBtn = document.getElementById('sidebar-add-topic');
    if (manageBtn) {
      manageBtn.style.display = authed ? '' : 'none';
    }
    // Show/hide note form on paper pages
    var noteForm = document.getElementById('note-form');
    if (noteForm) {
      noteForm.style.display = authed ? '' : 'none';
    }
    // Notify other modules about auth state change
    window.dispatchEvent(new CustomEvent('kb-auth-changed', { detail: { authenticated: authed } }));
  }

  function openModal() {
    passwordInput.value = '';
    modal.style.display = 'flex';
    passwordInput.focus();
  }

  function closeModal() {
    modal.style.display = 'none';
  }

  authBtn.addEventListener('click', openModal);
  cancelBtn.addEventListener('click', closeModal);
  overlay.addEventListener('click', closeModal);

  var WORKER_URL = 'https://frieren-lab-proxy.ntufrierenlab.workers.dev';

  function verifyAndSave() {
    var pwd = passwordInput.value.trim();
    if (!pwd) return;

    saveBtn.disabled = true;
    saveBtn.textContent = 'Verifying...';

    fetch(WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: pwd, action: 'status' })
    })
    .then(function (r) {
      if (r.status === 401) {
        throw new Error('Invalid password');
      }
      return r.json();
    })
    .then(function () {
      sessionStorage.setItem('kb-session-pwd', pwd);
      closeModal();
      updateAuthState();
    })
    .catch(function (err) {
      passwordInput.value = '';
      passwordInput.placeholder = err.message === 'Invalid password' ? 'Wrong password, try again...' : 'Error, try again...';
      passwordInput.focus();
    })
    .finally(function () {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save';
    });
  }

  saveBtn.addEventListener('click', verifyAndSave);

  passwordInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      verifyAndSave();
    }
  });

  updateAuthState();
})();

// Mobile sidebar toggle
(function () {
  var sidebar = document.getElementById('sidebar');
  var openBtn = document.getElementById('sidebar-toggle');
  var mobileToggle = document.getElementById('mobile-sidebar-toggle');
  var closeBtn = document.getElementById('sidebar-close');
  if (openBtn && sidebar) {
    openBtn.addEventListener('click', function () {
      sidebar.classList.add('open');
    });
  }
  if (mobileToggle && sidebar) {
    mobileToggle.addEventListener('click', function () {
      sidebar.classList.add('open');
    });
  }
  if (closeBtn && sidebar) {
    closeBtn.addEventListener('click', function () {
      sidebar.classList.remove('open');
    });
  }
})();

// Mobile top bar: move notification bell into top bar on mobile
(function () {
  var topbarActions = document.getElementById('mobile-topbar-actions');
  var notifBell = document.getElementById('notification-bell');
  if (topbarActions && notifBell && window.innerWidth <= 768) {
    topbarActions.appendChild(notifBell);
  }
})();

// Language toggle (EN/ZH)
(function () {
  var btn = document.getElementById('lang-toggle');
  if (!btn) return;

  var lang = localStorage.getItem('paper-lang') || 'en';
  applyLang(lang);

  btn.addEventListener('click', function () {
    lang = lang === 'en' ? 'zh' : 'en';
    localStorage.setItem('paper-lang', lang);
    applyLang(lang);
  });

  function applyLang(l) {
    var enEls = document.querySelectorAll('.lang-en');
    var zhEls = document.querySelectorAll('.lang-zh');
    var label = btn.querySelector('.lang-label');

    if (l === 'zh') {
      enEls.forEach(function (el) { el.style.display = 'none'; });
      zhEls.forEach(function (el) { el.style.display = ''; });
      label.textContent = 'EN';
      btn.setAttribute('data-lang', 'zh');
    } else {
      enEls.forEach(function (el) { el.style.display = ''; });
      zhEls.forEach(function (el) { el.style.display = 'none'; });
      label.textContent = '中文';
      btn.setAttribute('data-lang', 'en');
    }
  }
})();

// Digest language toggle (homepage)
(function () {
  var btn = document.getElementById('digest-lang-toggle');
  if (!btn) return;

  var lang = 'en';
  btn.addEventListener('click', function () {
    lang = lang === 'en' ? 'zh' : 'en';
    var label = btn.querySelector('.lang-label');
    var container = document.querySelector('.latest-digest-list');
    if (!container) return;
    var enEls = container.querySelectorAll('.lang-en');
    var zhEls = container.querySelectorAll('.lang-zh');
    if (lang === 'zh') {
      enEls.forEach(function (el) { el.style.display = 'none'; });
      zhEls.forEach(function (el) { el.style.display = ''; });
      label.textContent = 'EN';
      btn.setAttribute('data-lang', 'zh');
    } else {
      enEls.forEach(function (el) { el.style.display = ''; });
      zhEls.forEach(function (el) { el.style.display = 'none'; });
      label.textContent = '中文';
      btn.setAttribute('data-lang', 'en');
    }
  });
})();

// Sidebar topic management (shared across all pages)
(function () {
  var DEFAULT_TOPICS = ['Auto White Balance'];

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

  function addTopicItem(name) {
    var trimmed = name.trim();
    if (!trimmed) return false;
    var topics = getTopics();
    var exists = topics.some(function (t) { return t.toLowerCase() === trimmed.toLowerCase(); });
    if (exists) return false;
    topics.push(trimmed);
    saveTopics(topics);
    return true;
  }

  function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  var editMode = false;

  function removeTopic(name) {
    var topics = getTopics().filter(function (t) { return t !== name; });
    saveTopics(topics);
  }

  function refreshSidebarTopics() {
    var list = document.getElementById('sidebar-topic-list');
    if (!list) return;

    var hugoTopics = {};
    list.querySelectorAll('li[data-hugo-topic]').forEach(function (li) {
      hugoTopics[li.getAttribute('data-hugo-topic')] = li;
    });

    list.querySelectorAll('li[data-js-topic]').forEach(function (li) { li.remove(); });

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

    // Toggle editing class and delete buttons
    if (editMode) {
      list.classList.add('topic-editing');
    } else {
      list.classList.remove('topic-editing');
    }

    // Add or remove delete buttons
    var allItems = list.querySelectorAll('li');
    allItems.forEach(function (li) {
      var existing = li.querySelector('.topic-delete-btn');
      var topicName = li.getAttribute('data-hugo-topic') || li.getAttribute('data-js-topic');
      if (editMode && topicName) {
        if (!existing) {
          var anchor = li.querySelector('a');
          var countEl = li.querySelector('.topic-count');
          var count = countEl ? parseInt(countEl.textContent, 10) || 0 : 0;
          var btn = document.createElement('button');
          btn.className = 'topic-delete-btn' + (count > 0 ? ' disabled' : '');
          btn.setAttribute('data-topic', topicName);
          btn.title = count > 0 ? 'Remove all papers first' : 'Remove topic';
          btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
          btn.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            var name = this.getAttribute('data-topic');
            var li = this.closest('li');
            var cEl = li ? li.querySelector('.topic-count') : null;
            var c = cEl ? parseInt(cEl.textContent, 10) || 0 : 0;
            if (c > 0 || getTopics().length <= 1) return;
            removeTopic(name);
            refreshSidebarTopics();
          });
          // Insert inside the <a> tag, after .topic-count
          if (anchor) {
            anchor.appendChild(btn);
          } else {
            li.appendChild(btn);
          }
        }
      } else if (existing) {
        existing.remove();
      }
    });
  }

  // Sidebar manage-topic button: toggles edit mode
  var sidebarAddBtn = document.getElementById('sidebar-add-topic');
  var sidebarTopicForm = document.getElementById('sidebar-topic-form');
  var sidebarTopicInput = document.getElementById('sidebar-topic-input');
  var sidebarTopicSave = document.getElementById('sidebar-topic-save');
  var sidebarTopicCancel = document.getElementById('sidebar-topic-cancel');

  if (sidebarAddBtn) {
    sidebarAddBtn.addEventListener('click', function () {
      editMode = !editMode;
      sidebarAddBtn.classList.toggle('active', editMode);
      if (editMode) {
        sidebarTopicForm.style.display = 'block';
        sidebarTopicInput.value = '';
        sidebarTopicInput.focus();
      } else {
        sidebarTopicForm.style.display = 'none';
      }
      refreshSidebarTopics();
    });
  }
  if (sidebarTopicCancel) {
    sidebarTopicCancel.addEventListener('click', function () {
      editMode = false;
      sidebarAddBtn.classList.remove('active');
      sidebarTopicForm.style.display = 'none';
      refreshSidebarTopics();
    });
  }
  if (sidebarTopicSave) {
    sidebarTopicSave.addEventListener('click', function () {
      var name = sidebarTopicInput.value.trim();
      if (name && addTopicItem(name)) {
        sidebarTopicInput.value = '';
        sidebarTopicInput.focus();
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

  // Exit edit mode when auth is revoked
  window.addEventListener('kb-auth-changed', function (e) {
    if (!e.detail.authenticated && editMode) {
      editMode = false;
      if (sidebarAddBtn) sidebarAddBtn.classList.remove('active');
      if (sidebarTopicForm) sidebarTopicForm.style.display = 'none';
      refreshSidebarTopics();
    }
  });

  refreshSidebarTopics();
})();

// Simple client-side search
(function () {
  var input = document.getElementById('search-input');
  if (!input) return;

  var resultsDiv = document.createElement('div');
  resultsDiv.className = 'search-results';
  input.parentNode.appendChild(resultsDiv);

  var papers = [];
  fetch('/index.json')
    .then(function (r) { return r.json(); })
    .then(function (data) { papers = data; })
    .catch(function () {});

  input.addEventListener('input', function () {
    var q = this.value.toLowerCase().trim();
    resultsDiv.innerHTML = '';
    if (q.length < 2) {
      resultsDiv.classList.remove('active');
      return;
    }
    var matches = papers.filter(function (p) {
      return p.title.toLowerCase().includes(q) ||
        (p.summary && p.summary.toLowerCase().includes(q)) ||
        (p.tags && p.tags.some(function (t) { return t.toLowerCase().includes(q); }));
    }).slice(0, 8);

    if (matches.length === 0) {
      resultsDiv.classList.remove('active');
      return;
    }
    matches.forEach(function (p) {
      var item = document.createElement('div');
      item.className = 'search-result-item';
      item.innerHTML = '<a href="' + p.url + '">' + p.title + '</a>' +
        '<div class="search-result-date">' + (p.date || '') + '</div>';
      resultsDiv.appendChild(item);
    });
    resultsDiv.classList.add('active');
  });

  document.addEventListener('click', function (e) {
    if (!input.contains(e.target) && !resultsDiv.contains(e.target)) {
      resultsDiv.classList.remove('active');
    }
  });
})();

// ── Notification Bell System ─────────────────────────────────────
(function () {
  var WORKER_URL = 'https://frieren-lab-proxy.ntufrierenlab.workers.dev';
  var POLL_INTERVAL = 20000;
  var MATCH_WINDOW = 120000;
  var pollTimer = null;

  var bellContainer = document.getElementById('notification-bell');
  var bellBtn = document.getElementById('notif-bell-btn');
  var badge = document.getElementById('notif-badge');
  var dropdown = document.getElementById('notif-dropdown');
  var dropdownBody = document.getElementById('notif-dropdown-body');
  var clearBtn = document.getElementById('notif-clear-btn');
  if (!bellContainer) return;

  // ── State via sessionStorage ────────────────────────────────────
  function getPapers() {
    try { return JSON.parse(sessionStorage.getItem('kb-pending-papers') || '[]'); }
    catch (e) { return []; }
  }

  function savePapers(papers) {
    sessionStorage.setItem('kb-pending-papers', JSON.stringify(papers));
  }

  function getPassword() {
    return sessionStorage.getItem('kb-session-pwd') || '';
  }

  // ── Bell visibility ─────────────────────────────────────────────
  function updateBellVisibility() {
    bellContainer.style.display = getPapers().length > 0 ? '' : 'none';
  }

  // ── Badge ───────────────────────────────────────────────────────
  function updateBadge() {
    var unread = getPapers().filter(function (p) {
      return !p.readByUser && (p.status === 'completed' || p.status === 'failed');
    }).length;
    if (unread > 0) {
      badge.textContent = unread;
      badge.style.display = '';
    } else {
      badge.style.display = 'none';
    }
  }

  // ── Render dropdown items ───────────────────────────────────────
  function renderDropdown() {
    var papers = getPapers();
    dropdownBody.innerHTML = '';

    if (papers.length === 0) {
      dropdownBody.innerHTML = '<div class="notif-empty">No pending papers</div>';
      return;
    }

    papers.slice().reverse().forEach(function (paper) {
      var item = document.createElement('div');
      item.className = 'notif-item';

      var iconClass = paper.status;
      var iconContent = '';
      var statusText = '';

      var isDelete = paper.type === 'delete';

      if (paper.status === 'pending') {
        iconContent = '<div class="notif-spinner"></div>';
        statusText = isDelete ? 'Queued for deletion...' : 'Queued...';
      } else if (paper.status === 'processing') {
        iconContent = '<div class="notif-spinner"></div>';
        statusText = isDelete ? 'Deleting paper...' : 'Generating summary...';
      } else if (paper.status === 'completed') {
        iconContent = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>';
        statusText = isDelete ? 'Paper deleted!' : 'Paper is live!';
      } else if (paper.status === 'failed') {
        iconContent = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
        statusText = isDelete ? 'Delete failed' : 'Workflow failed';
      }

      var timeAgo = relativeTime(paper.triggeredAt);
      var safeTitle = escapeHtml(paper.title.length > 60 ? paper.title.substring(0, 60) + '...' : paper.title);

      item.innerHTML =
        '<div class="notif-item-icon ' + iconClass + '">' + iconContent + '</div>' +
        '<div class="notif-item-content">' +
          '<div class="notif-item-title">' + safeTitle + '</div>' +
          '<div class="notif-item-status">' + statusText + '</div>' +
          '<div class="notif-item-time">' + timeAgo + '</div>' +
        '</div>';

      dropdownBody.appendChild(item);
    });
  }

  // ── Dropdown toggle ─────────────────────────────────────────────
  var isOpen = false;

  function openDropdown() {
    renderDropdown();
    dropdown.style.display = '';
    isOpen = true;

    // Mark all terminal items as read
    var papers = getPapers();
    var changed = false;
    papers.forEach(function (p) {
      if (!p.readByUser && (p.status === 'completed' || p.status === 'failed')) {
        p.readByUser = true;
        changed = true;
      }
    });
    if (changed) {
      savePapers(papers);
      updateBadge();
    }
  }

  function closeDropdown() {
    dropdown.style.display = 'none';
    isOpen = false;
  }

  bellBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    isOpen ? closeDropdown() : openDropdown();
  });

  document.addEventListener('click', function (e) {
    if (isOpen && !bellContainer.contains(e.target)) closeDropdown();
  });

  clearBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    sessionStorage.removeItem('kb-pending-papers');
    stopPolling();
    updateBellVisibility();
    updateBadge();
    closeDropdown();
  });

  // ── Polling ─────────────────────────────────────────────────────
  function startPolling() {
    if (pollTimer) return;
    pollStatus();
    pollTimer = setInterval(pollStatus, POLL_INTERVAL);
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  function pollStatus() {
    var password = getPassword();
    if (!password) return;

    var papers = getPapers();
    var hasActive = papers.some(function (p) {
      return p.status === 'pending' || p.status === 'processing';
    });
    if (!hasActive) { stopPolling(); return; }

    fetch(WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: password, action: 'status' })
    })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (!data.ok || !data.runs) return;
      var runs = data.runs;
      var papers = getPapers();
      var changed = false;

      // Collect already-matched run IDs
      var claimed = {};
      papers.forEach(function (p) { if (p.runId) claimed[p.runId] = true; });

      papers.forEach(function (paper) {
        if (paper.status === 'completed' || paper.status === 'failed') return;

        // Already matched — update from that run
        if (paper.runId) {
          var run = runs.find(function (r) { return r.id === paper.runId; });
          if (run) {
            var s = mapStatus(run);
            if (s !== paper.status) { paper.status = s; changed = true; }
          }
          return;
        }

        // Not yet matched — find by timestamp
        var triggerMs = new Date(paper.triggeredAt).getTime();
        var best = null;
        var bestDiff = Infinity;
        runs.forEach(function (run) {
          if (claimed[run.id]) return;
          var diff = new Date(run.created_at).getTime() - triggerMs;
          if (diff >= -5000 && diff <= MATCH_WINDOW && diff < bestDiff) {
            bestDiff = diff;
            best = run;
          }
        });

        if (best) {
          paper.runId = best.id;
          claimed[best.id] = true;
          paper.status = mapStatus(best);
          changed = true;
        }
      });

      if (changed) {
        savePapers(papers);
        updateBadge();
        updateBellVisibility();
        if (isOpen) renderDropdown();

        // Show refresh banner if any paper just completed
        var hasNewlyCompleted = papers.some(function (p) {
          return p.status === 'completed' && !p.readByUser;
        });
        if (hasNewlyCompleted) showRefreshBanner();
      }

      if (!papers.some(function (p) { return p.status === 'pending' || p.status === 'processing'; })) {
        stopPolling();
      }
    })
    .catch(function () {});
  }

  function mapStatus(run) {
    if (run.status === 'completed') return run.conclusion === 'success' ? 'completed' : 'failed';
    if (run.status === 'in_progress') return 'processing';
    return 'pending';
  }

  // ── Helpers ─────────────────────────────────────────────────────
  function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function relativeTime(iso) {
    var sec = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
    if (sec < 60) return 'Just now';
    var min = Math.floor(sec / 60);
    if (min < 60) return min + 'm ago';
    var hr = Math.floor(min / 60);
    if (hr < 24) return hr + 'h ago';
    return Math.floor(hr / 24) + 'd ago';
  }

  // ── Refresh banner ──────────────────────────────────────────────
  function showRefreshBanner() {
    if (document.getElementById('kb-refresh-banner')) return;
    var banner = document.createElement('div');
    banner.id = 'kb-refresh-banner';
    banner.className = 'refresh-banner';
    banner.innerHTML =
      '<div class="refresh-banner-content">' +
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>' +
        '<span>Site has been updated — <a href="javascript:location.reload()">Refresh</a> to see new content</span>' +
        '<button class="refresh-banner-close" aria-label="Dismiss">' +
          '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>' +
        '</button>' +
      '</div>';
    document.body.appendChild(banner);
    banner.querySelector('.refresh-banner-close').addEventListener('click', function () {
      banner.remove();
    });
    // Auto-show with slide animation
    requestAnimationFrame(function () {
      banner.classList.add('visible');
    });
  }

  // ── Listen for new papers from search page ──────────────────────
  window.addEventListener('kb-paper-added', function () {
    updateBellVisibility();
    updateBadge();
    startPolling();
    if (isOpen) renderDropdown();
  });

  // ── Init ────────────────────────────────────────────────────────
  updateBellVisibility();
  updateBadge();
  if (getPapers().some(function (p) { return p.status === 'pending' || p.status === 'processing'; })) {
    startPolling();
  }
})();

// ── Delete Paper ─────────────────────────────────────────────────
(function () {
  var WORKER_URL = 'https://frieren-lab-proxy.ntufrierenlab.workers.dev';

  function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  var deleteBtn = document.getElementById('btn-delete-paper');
  if (!deleteBtn) return;

  var modal = document.getElementById('delete-confirm-modal');
  var modalOverlay = document.getElementById('delete-modal-overlay');
  var confirmBtn = document.getElementById('delete-confirm');
  var cancelBtn = document.getElementById('delete-cancel');
  var titleEl = document.getElementById('delete-confirm-title');

  function openModal() {
    titleEl.textContent = deleteBtn.getAttribute('data-title');
    modal.style.display = '';
  }

  function closeModal() {
    modal.style.display = 'none';
  }

  deleteBtn.addEventListener('click', openModal);
  cancelBtn.addEventListener('click', closeModal);
  modalOverlay.addEventListener('click', closeModal);

  confirmBtn.addEventListener('click', function () {
    var filename = deleteBtn.getAttribute('data-filename');
    var title = deleteBtn.getAttribute('data-title');
    var password = sessionStorage.getItem('kb-session-pwd');

    if (!password) {
      closeModal();
      return;
    }

    // Disable button to prevent double-click
    confirmBtn.disabled = true;
    confirmBtn.textContent = 'Deleting...';

    fetch(WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        password: password,
        action: 'delete',
        paper_filename: filename
      })
    })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      closeModal();
      confirmBtn.disabled = false;
      confirmBtn.textContent = 'Delete';

      if (data.ok) {
        // Add to pending papers for notification tracking
        var papers = [];
        try { papers = JSON.parse(sessionStorage.getItem('kb-pending-papers') || '[]'); }
        catch (e) { papers = []; }

        papers.push({
          title: title,
          type: 'delete',
          triggeredAt: new Date().toISOString(),
          status: 'pending',
          runId: null,
          readByUser: false
        });
        sessionStorage.setItem('kb-pending-papers', JSON.stringify(papers));

        // Notify bell system
        window.dispatchEvent(new CustomEvent('kb-paper-added'));

        // Replace page content with "deleted" confirmation
        var article = document.querySelector('.paper-detail');
        if (article) {
          article.innerHTML =
            '<div class="paper-deleted-notice">' +
              '<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">' +
                '<polyline points="3 6 5 6 21 6"/>' +
                '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>' +
              '</svg>' +
              '<h2>Paper Deleted</h2>' +
              '<p class="deleted-title">' + escapeHtml(title) + '</p>' +
              '<p class="deleted-desc">The deletion is being processed. The site will update shortly.</p>' +
              '<a href="/" class="btn btn-primary">Back to Home</a>' +
            '</div>';
        }
        var delModal = document.getElementById('delete-confirm-modal');
        if (delModal) delModal.style.display = 'none';
      } else {
        alert('Failed to trigger delete: ' + (data.error || 'Unknown error'));
      }
    })
    .catch(function (err) {
      closeModal();
      confirmBtn.disabled = false;
      confirmBtn.textContent = 'Delete';
      alert('Network error: ' + err.message);
    });
  });
})();

// ── Notes (Add & Delete) ──────────────────────────────────────────
(function () {
  var WORKER_URL = 'https://frieren-lab-proxy.ntufrierenlab.workers.dev';

  var notesSection = document.getElementById('paper-notes');
  var noteInput = document.getElementById('note-input');
  var noteSubmit = document.getElementById('note-submit');
  var notesList = document.getElementById('notes-list');
  var notesCount = document.getElementById('notes-count');
  if (!notesSection || !noteInput || !noteSubmit) return;

  var paperFilename = notesSection.getAttribute('data-filename');

  function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function formatDate(date) {
    var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    var h = date.getHours();
    var m = date.getMinutes();
    var ampm = h >= 12 ? 'PM' : 'AM';
    h = h % 12 || 12;
    var mm = m < 10 ? '0' + m : m;
    return months[date.getMonth()] + ' ' + date.getDate() + ', ' + date.getFullYear() + ' · ' + h + ':' + mm + ' ' + ampm;
  }

  function updateNoteCount() {
    var count = notesList.querySelectorAll('.note-item').length;
    notesCount.textContent = count > 0 ? '(' + count + ')' : '';
  }

  function trackPending(title, type) {
    var pending = [];
    try { pending = JSON.parse(sessionStorage.getItem('kb-pending-papers') || '[]'); } catch (e) { pending = []; }
    pending.push({
      title: title,
      type: type,
      triggeredAt: new Date().toISOString(),
      status: 'pending',
      runId: null,
      readByUser: false
    });
    sessionStorage.setItem('kb-pending-papers', JSON.stringify(pending));
    window.dispatchEvent(new CustomEvent('kb-paper-added'));
  }

  // ── Add note ──
  function submitNote() {
    var text = noteInput.value.trim();
    if (!text) return;

    var password = sessionStorage.getItem('kb-session-pwd');
    if (!password) return;

    noteSubmit.disabled = true;
    noteSubmit.textContent = 'Submitting...';

    fetch(WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        password: password,
        action: 'add-note',
        paper_filename: paperFilename,
        note_text: text
      })
    })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data.ok) {
        var empty = document.getElementById('notes-empty');
        if (empty) empty.remove();

        var noteItem = document.createElement('div');
        noteItem.className = 'note-item note-new';
        var now = new Date();
        noteItem.setAttribute('data-date', now.toISOString());
        noteItem.innerHTML =
          '<div class="note-body"><p>' + escapeHtml(text) + '</p></div>' +
          '<div class="note-meta">' +
            '<span>' + formatDate(now) + '</span>' +
            '<button class="note-delete-btn visible" title="Delete note">' +
              '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>' +
            '</button>' +
          '</div>';
        notesList.appendChild(noteItem);
        bindDeleteBtn(noteItem.querySelector('.note-delete-btn'));

        updateNoteCount();
        noteInput.value = '';
        trackPending('Note added', 'note');
      } else {
        throw new Error(data.error || 'Failed to add note');
      }
    })
    .catch(function (err) {
      alert('Error: ' + err.message);
    })
    .finally(function () {
      noteSubmit.disabled = false;
      noteSubmit.textContent = 'Add Note';
    });
  }

  noteSubmit.addEventListener('click', submitNote);
  noteInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      submitNote();
    }
  });

  // ── Delete note ──
  function deleteNote(btn) {
    var noteItem = btn.closest('.note-item');
    var noteDate = noteItem.getAttribute('data-date');
    var password = sessionStorage.getItem('kb-session-pwd');
    if (!password || !noteDate) return;

    if (!confirm('Delete this note?')) return;

    btn.disabled = true;

    fetch(WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        password: password,
        action: 'delete-note',
        paper_filename: paperFilename,
        note_date: noteDate
      })
    })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data.ok) {
        noteItem.style.opacity = '0';
        noteItem.style.transition = 'opacity 0.3s';
        setTimeout(function () {
          noteItem.remove();
          updateNoteCount();
          if (notesList.querySelectorAll('.note-item').length === 0) {
            var p = document.createElement('p');
            p.className = 'notes-empty';
            p.id = 'notes-empty';
            p.textContent = 'No notes yet.';
            notesList.appendChild(p);
          }
        }, 300);
        trackPending('Note deleted', 'note');
      } else {
        throw new Error(data.error || 'Failed to delete note');
      }
    })
    .catch(function (err) {
      btn.disabled = false;
      alert('Error: ' + err.message);
    });
  }

  function bindDeleteBtn(btn) {
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      deleteNote(this);
    });
  }

  // Bind existing delete buttons
  notesList.querySelectorAll('.note-delete-btn').forEach(bindDeleteBtn);

  // ── Toggle delete buttons on auth change ──
  function toggleDeleteBtns(show) {
    notesList.querySelectorAll('.note-delete-btn').forEach(function (btn) {
      if (show) {
        btn.classList.add('visible');
      } else {
        btn.classList.remove('visible');
      }
    });
  }

  window.addEventListener('kb-auth-changed', function (e) {
    toggleDeleteBtns(e.detail.authenticated);
  });

  // Set initial state
  if (sessionStorage.getItem('kb-session-pwd')) {
    toggleDeleteBtns(true);
  }
})();
