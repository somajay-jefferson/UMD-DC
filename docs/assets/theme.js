(function () {
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem('theme');
    if (saved === 'light' || saved === 'dark') root.setAttribute('data-theme', saved);
  } catch (e) {}
  function current() {
    var a = root.getAttribute('data-theme');
    if (a) return a;
    return window.matchMedia &&
           window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  var btn = document.getElementById('themetoggle');
  if (!btn) return;
  function label() { btn.textContent = current() === 'dark' ? 'Light' : 'Dark'; }
  label();
  btn.addEventListener('click', function () {
    var next = current() === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('theme', next); } catch (e) {}
    label();
  });
})();
