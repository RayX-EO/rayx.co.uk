(function(){
  const header = document.querySelector('.site-header');
  const setHeaderState = () => {
    if(!header) return;
    header.classList.toggle('is-scrolled', window.scrollY > 8);
  };
  setHeaderState();
  window.addEventListener('scroll', setHeaderState, { passive: true });

  const toggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('.nav-links');

  const setNavOpen = (open) => {
    if(!toggle || !nav) return;
    nav.classList.toggle('open', open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
  };

  if(toggle && nav){
    setNavOpen(false);

    toggle.addEventListener('click', () => {
      setNavOpen(!nav.classList.contains('open'));
    });

    nav.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => setNavOpen(false));
    });

    document.addEventListener('keydown', (e) => {
      if(e.key === 'Escape') setNavOpen(false);
    });

    document.addEventListener('click', (e) => {
      if(!nav.classList.contains('open')) return;
      if(!(e.target instanceof Element)) return;
      if(nav.contains(e.target) || toggle.contains(e.target)) return;
      setNavOpen(false);
    });
  }

  const y = document.getElementById('year');
  if(y) y.textContent = new Date().getFullYear();
})();
