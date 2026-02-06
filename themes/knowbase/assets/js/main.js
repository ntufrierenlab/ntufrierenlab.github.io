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

// Mobile sidebar toggle
(function () {
  var sidebar = document.getElementById('sidebar');
  var openBtn = document.getElementById('sidebar-toggle');
  var closeBtn = document.getElementById('sidebar-close');
  if (openBtn && sidebar) {
    openBtn.addEventListener('click', function () {
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
  var DEFAULT_TOPICS = ['Auto White Balance'];

  function getTopics() {
    var stored = localStorage.getItem('kb-topics');
    if (stored) {
      try { return JSON.parse(stored); } catch (e) { /* ignore */ }
    }
    localStorage.setItem('kb-topics', JSON.stringify(DEFAULT_TOPICS));
    return DEFAULT_TOPICS.slice();
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
      if (name && addTopicItem(name)) {
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
