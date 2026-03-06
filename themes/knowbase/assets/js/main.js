// ── Shared Utilities (KB namespace) ──────────────────────────────
window.KB = (function () {
  var DEFAULT_TOPICS = ['Auto White Balance'];

  function escapeHtml(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function escapeAttr(s) {
    return s.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function getTopics() {
    var stored = localStorage.getItem('kb-topics');
    var topics;
    if (stored) {
      try { topics = JSON.parse(stored); } catch (e) { topics = DEFAULT_TOPICS.slice(); }
    } else {
      topics = DEFAULT_TOPICS.slice();
    }
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

  // Toast notification system
  var toastContainer = null;

  function getToastContainer() {
    if (!toastContainer) {
      toastContainer = document.createElement('div');
      toastContainer.className = 'kb-toast-container';
      document.body.appendChild(toastContainer);
    }
    return toastContainer;
  }

  function showToast(message, type) {
    type = type || 'info';
    var container = getToastContainer();
    var toast = document.createElement('div');
    toast.className = 'kb-toast kb-toast-' + type;

    var icons = {
      error: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
      success: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="16 9 10.5 14.5 8 12"/></svg>',
      info: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
    };

    toast.innerHTML =
      '<span class="kb-toast-icon">' + (icons[type] || icons.info) + '</span>' +
      '<span class="kb-toast-msg">' + escapeHtml(message) + '</span>' +
      '<button class="kb-toast-close" aria-label="Dismiss">' +
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>' +
      '</button>';

    container.appendChild(toast);

    var closeBtn = toast.querySelector('.kb-toast-close');
    function dismiss() {
      toast.classList.add('kb-toast-exit');
      setTimeout(function () { toast.remove(); }, 300);
    }
    closeBtn.addEventListener('click', dismiss);

    // Auto-dismiss after 5s
    setTimeout(dismiss, 5000);

    return toast;
  }

  return {
    escapeHtml: escapeHtml,
    escapeAttr: escapeAttr,
    getTopics: getTopics,
    saveTopics: saveTopics,
    showToast: showToast
  };
})();

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

// Sidebar topic management (shared across all pages)
(function () {
  function addTopicItem(name) {
    var trimmed = name.trim();
    if (!trimmed) return false;
    var topics = KB.getTopics();
    var exists = topics.some(function (t) { return t.toLowerCase() === trimmed.toLowerCase(); });
    if (exists) return false;
    topics.push(trimmed);
    KB.saveTopics(topics);
    return true;
  }

  var editMode = false;

  function removeTopic(name) {
    var topics = KB.getTopics().filter(function (t) { return t !== name; });
    KB.saveTopics(topics);
  }

  function refreshSidebarTopics() {
    var list = document.getElementById('sidebar-topic-list');
    if (!list) return;

    var hugoTopics = {};
    list.querySelectorAll('li[data-hugo-topic]').forEach(function (li) {
      hugoTopics[li.getAttribute('data-hugo-topic')] = li;
    });

    list.querySelectorAll('li[data-js-topic]').forEach(function (li) { li.remove(); });

    var topics = KB.getTopics();
    topics.forEach(function (t) {
      if (!hugoTopics[t]) {
        var li = document.createElement('li');
        li.setAttribute('data-js-topic', t);
        li.innerHTML =
          '<a href="javascript:void(0)">' +
            '<span class="topic-dot"></span>' +
            KB.escapeHtml(t) +
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
            if (c > 0 || KB.getTopics().length <= 1) return;
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
      var isTopicUpdate = paper.type === 'update-topics';

      if (paper.status === 'pending') {
        iconContent = '<div class="notif-spinner"></div>';
        statusText = isDelete ? 'Queued for deletion...' : isTopicUpdate ? 'Updating topics...' : 'Queued...';
      } else if (paper.status === 'processing') {
        iconContent = '<div class="notif-spinner"></div>';
        statusText = isDelete ? 'Deleting paper...' : isTopicUpdate ? 'Updating topics...' : 'Generating summary...';
      } else if (paper.status === 'deploying') {
        iconContent = '<div class="notif-spinner"></div>';
        statusText = 'Deploying site...';
      } else if (paper.status === 'completed') {
        iconContent = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>';
        statusText = isDelete ? 'Paper deleted!' : isTopicUpdate ? 'Topics updated!' : 'Paper is live!';
      } else if (paper.status === 'failed') {
        iconContent = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
        statusText = isDelete ? 'Delete failed' : isTopicUpdate ? 'Topic update failed' : 'Workflow failed';
      }

      var timeAgo = relativeTime(paper.triggeredAt);
      var safeTitle = KB.escapeHtml(paper.title.length > 60 ? paper.title.substring(0, 60) + '...' : paper.title);

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
      return p.status === 'pending' || p.status === 'processing' || p.status === 'deploying';
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
      var deployRuns = data.deployRuns || [];
      var papers = getPapers();
      var changed = false;

      // Collect already-matched run IDs
      var claimed = {};
      papers.forEach(function (p) { if (p.runId) claimed[p.runId] = true; });

      papers.forEach(function (paper) {
        if (paper.status === 'completed' || paper.status === 'failed') return;

        // Phase 2: waiting for deploy
        if (paper.status === 'deploying') {
          var afterMs = new Date(paper.deployAfter).getTime();
          var deployRun = null;
          deployRuns.forEach(function (dr) {
            if (new Date(dr.created_at).getTime() >= afterMs - 5000) {
              if (!deployRun || new Date(dr.created_at) > new Date(deployRun.created_at))
                deployRun = dr;
            }
          });
          if (deployRun && deployRun.status === 'completed') {
            paper.status = deployRun.conclusion === 'success' ? 'completed' : 'failed';
            changed = true;
          }
          return;
        }

        // Phase 1: action workflow matching
        // Already matched — update from that run
        if (paper.runId) {
          var run = runs.find(function (r) { return r.id === paper.runId; });
          if (run) {
            var s = mapStatus(run);
            if (s === 'completed') {
              paper.status = 'deploying';
              paper.deployAfter = run.updated_at;
              changed = true;
            } else if (s !== paper.status) {
              paper.status = s;
              changed = true;
            }
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
          var bestStatus = mapStatus(best);
          if (bestStatus === 'completed') {
            paper.status = 'deploying';
            paper.deployAfter = best.updated_at;
          } else {
            paper.status = bestStatus;
          }
          changed = true;
        }
      });

      if (changed) {
        savePapers(papers);
        updateBadge();
        updateBellVisibility();
        if (isOpen) renderDropdown();

        // Show refresh banner + dismiss processing toast if any paper just completed
        var hasNewlyCompleted = papers.some(function (p) {
          return p.status === 'completed' && !p.readByUser;
        });
        if (hasNewlyCompleted) {
          showRefreshBanner();
          var processToast = document.getElementById('process-toast');
          if (processToast) processToast.style.display = 'none';
        }
      }

      if (!papers.some(function (p) { return p.status === 'pending' || p.status === 'processing' || p.status === 'deploying'; })) {
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
  if (getPapers().some(function (p) { return p.status === 'pending' || p.status === 'processing' || p.status === 'deploying'; })) {
    startPolling();
  }
})();

// ── Edit Topics ──────────────────────────────────────────────────
(function () {
  var WORKER_URL = 'https://frieren-lab-proxy.ntufrierenlab.workers.dev';

  var editBtn = document.getElementById('btn-edit-topics');
  if (!editBtn) return;

  var modal = document.getElementById('edit-topics-modal');
  var overlay = document.getElementById('edit-topics-overlay');
  var topicListEl = document.getElementById('edit-topics-list');
  var newInput = document.getElementById('edit-topics-new-input');
  var newAddBtn = document.getElementById('edit-topics-new-add');
  var cancelBtn = document.getElementById('edit-topics-cancel');
  var saveBtn = document.getElementById('edit-topics-save');

  var currentTopics = editBtn.getAttribute('data-topics').split(',').map(function (t) { return t.trim(); }).filter(Boolean);

  function renderTopicList() {
    var topics = KB.getTopics();
    topicListEl.innerHTML = '';
    topics.forEach(function (t) {
      var label = document.createElement('label');
      label.className = 'topic-picker-item';
      var isChecked = currentTopics.some(function (ct) { return ct.toLowerCase() === t.toLowerCase(); });
      label.innerHTML =
        '<input type="checkbox" name="edit-topic-pick" value="' + KB.escapeAttr(t) + '"' + (isChecked ? ' checked' : '') + '>' +
        '<span class="topic-picker-check"></span>' +
        '<span class="topic-picker-name">' + KB.escapeHtml(t) + '</span>';
      topicListEl.appendChild(label);
    });
  }

  function openModal() {
    currentTopics = editBtn.getAttribute('data-topics').split(',').map(function (t) { return t.trim(); }).filter(Boolean);
    renderTopicList();
    newInput.value = '';
    modal.style.display = 'flex';
  }

  function closeModal() {
    modal.style.display = 'none';
  }

  editBtn.addEventListener('click', openModal);
  cancelBtn.addEventListener('click', closeModal);
  overlay.addEventListener('click', closeModal);

  newAddBtn.addEventListener('click', function () {
    var name = newInput.value.trim();
    if (!name) return;
    var topics = KB.getTopics();
    var exists = topics.some(function (t) { return t.toLowerCase() === name.toLowerCase(); });
    if (!exists) {
      topics.push(name);
      KB.saveTopics(topics);
    }
    currentTopics.push(name);
    newInput.value = '';
    renderTopicList();
  });

  newInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      newAddBtn.click();
    }
  });

  saveBtn.addEventListener('click', function () {
    var checked = topicListEl.querySelectorAll('input[type="checkbox"]:checked');
    var topics = [];
    checked.forEach(function (cb) { topics.push(cb.value); });
    if (topics.length === 0) {
      KB.showToast('Please select at least one topic.', 'error');
      return;
    }

    var password = sessionStorage.getItem('kb-session-pwd');
    if (!password) return;

    var filename = editBtn.getAttribute('data-filename');
    var topicStr = topics.join(',');

    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    fetch(WORKER_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        password: password,
        action: 'update-topics',
        paper_filename: filename,
        topics: topicStr
      })
    })
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (data.ok) {
        closeModal();
        // Update topic badges in the DOM
        var topicsDiv = document.querySelector('.paper-topics');
        if (topicsDiv) {
          // Remove old badges
          var oldBadges = topicsDiv.querySelectorAll('.topic-badge');
          oldBadges.forEach(function (b) { b.remove(); });
          // Insert new badges before the edit button
          topics.forEach(function (t) {
            var a = document.createElement('a');
            a.href = 'topics/' + t.toLowerCase().replace(/\s+/g, '-') + '/';
            a.className = 'topic-badge';
            a.textContent = t;
            topicsDiv.insertBefore(a, editBtn);
          });
        }
        // Update data attribute
        editBtn.setAttribute('data-topics', topicStr);

        // Track in notification system
        var pending = [];
        try { pending = JSON.parse(sessionStorage.getItem('kb-pending-papers') || '[]'); } catch (e) { pending = []; }
        pending.push({
          title: 'Topics updated',
          type: 'update-topics',
          triggeredAt: new Date().toISOString(),
          status: 'pending',
          runId: null,
          readByUser: false
        });
        sessionStorage.setItem('kb-pending-papers', JSON.stringify(pending));
        window.dispatchEvent(new CustomEvent('kb-paper-added'));
      } else {
        throw new Error(data.error || 'Failed to update topics');
      }
    })
    .catch(function (err) {
      KB.showToast('Error: ' + err.message, 'error');
    })
    .finally(function () {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save';
    });
  });

  // Show/hide edit button based on auth state
  window.addEventListener('kb-auth-changed', function (e) {
    editBtn.style.display = e.detail.authenticated ? '' : 'none';
  });

  // Set initial state
  if (sessionStorage.getItem('kb-session-pwd')) {
    editBtn.style.display = '';
  }
})();

// ── Delete Paper ─────────────────────────────────────────────────
(function () {
  var WORKER_URL = 'https://frieren-lab-proxy.ntufrierenlab.workers.dev';

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
              '<p class="deleted-title">' + KB.escapeHtml(title) + '</p>' +
              '<p class="deleted-desc">The deletion is being processed. The site will update shortly.</p>' +
              '<a href="/" class="btn btn-primary">Back to Home</a>' +
            '</div>';
        }
        var delModal = document.getElementById('delete-confirm-modal');
        if (delModal) delModal.style.display = 'none';
      } else {
        KB.showToast('Failed to trigger delete: ' + (data.error || 'Unknown error'), 'error');
      }
    })
    .catch(function (err) {
      closeModal();
      confirmBtn.disabled = false;
      confirmBtn.textContent = 'Delete';
      KB.showToast('Network error: ' + err.message, 'error');
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
          '<div class="note-body"><p>' + KB.escapeHtml(text) + '</p></div>' +
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
      KB.showToast('Error: ' + err.message, 'error');
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

    var noteDeleteModal = document.getElementById('note-delete-modal');
    var noteDeleteConfirm = document.getElementById('note-delete-confirm');
    var noteDeleteCancel = document.getElementById('note-delete-cancel');
    var noteDeleteOverlay = document.getElementById('note-delete-overlay');
    if (!noteDeleteModal) return;

    noteDeleteModal.style.display = 'flex';

    function cleanup() {
      noteDeleteModal.style.display = 'none';
      noteDeleteConfirm.removeEventListener('click', onConfirm);
      noteDeleteCancel.removeEventListener('click', onCancel);
      noteDeleteOverlay.removeEventListener('click', onCancel);
    }
    function onCancel() { cleanup(); }
    function onConfirm() {
      cleanup();
      doDeleteNote(btn, noteItem, noteDate, password);
    }
    noteDeleteConfirm.addEventListener('click', onConfirm);
    noteDeleteCancel.addEventListener('click', onCancel);
    noteDeleteOverlay.addEventListener('click', onCancel);
  }

  function doDeleteNote(btn, noteItem, noteDate, password) {
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
      KB.showToast('Error: ' + err.message, 'error');
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

// ── Escape Key Closes Modals ──────────────────────────────────────
(function () {
  var modalIds = ['note-delete-modal', 'delete-confirm-modal', 'edit-topics-modal', 'topic-modal', 'password-modal'];
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    for (var i = 0; i < modalIds.length; i++) {
      var modal = document.getElementById(modalIds[i]);
      if (modal && modal.style.display !== 'none' && modal.style.display !== '') {
        modal.style.display = 'none';
        e.preventDefault();
        return;
      }
    }
  });
})();

// ── Mobile Sidebar Auto-close on Nav Click ─────────────────────────
(function () {
  var sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  sidebar.addEventListener('click', function (e) {
    var link = e.target.closest('a');
    if (link && window.innerWidth <= 768) {
      sidebar.classList.remove('open');
    }
  });
})();

// ── Notification Bell Resize Handler ────────────────────────────────
(function () {
  var notifBell = document.getElementById('notification-bell');
  var topbarActions = document.getElementById('mobile-topbar-actions');
  if (!notifBell || !topbarActions) return;

  var originalParent = notifBell.parentNode;
  var originalNext = notifBell.nextSibling;
  var mql = window.matchMedia('(max-width: 768px)');

  function moveBell(mobile) {
    if (mobile) {
      topbarActions.appendChild(notifBell);
    } else if (notifBell.parentNode !== originalParent) {
      originalParent.insertBefore(notifBell, originalNext);
    }
  }

  moveBell(mql.matches);
  mql.addEventListener('change', function (e) {
    moveBell(e.matches);
  });
})();
