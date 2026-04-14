(function(){
  /* Mark JS enabled */
  document.documentElement.classList.add('js');

  /* Header scroll state */
  const header = document.querySelector('.site-header');
  const setHeaderState = () => {
    if(!header) return;
    header.classList.toggle('is-scrolled', window.scrollY > 8);
  };
  setHeaderState();
  window.addEventListener('scroll', setHeaderState, { passive: true });

  /* Mobile nav toggle */
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

  /* Footer year */
  const y = document.getElementById('year');
  if(y) y.textContent = new Date().getFullYear();

  /* Scroll-triggered reveal for content sections */
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if(!prefersReduced && 'IntersectionObserver' in window) {
    const revealTargets = document.querySelectorAll(
      '.proof-card, .workflow-stage, .audience-card, .row-panel, .layer-band, .story-band, .contact-note, .table-shell, .contact-form'
    );

    const revealStyle = document.createElement('style');
    revealStyle.textContent = `
      .reveal-ready {
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.5s cubic-bezier(0.16, 1, 0.3, 1),
                    transform 0.5s cubic-bezier(0.16, 1, 0.3, 1);
      }
      .reveal-visible {
        opacity: 1;
        transform: translateY(0);
      }
    `;
    document.head.appendChild(revealStyle);

    revealTargets.forEach((el, i) => {
      el.classList.add('reveal-ready');
      /* Stagger siblings slightly */
      const siblingIndex = Array.from(el.parentElement.children).indexOf(el);
      el.style.transitionDelay = (siblingIndex * 0.06) + 's';
    });

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if(entry.isIntersecting) {
          entry.target.classList.add('reveal-visible');
          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.08,
      rootMargin: '0px 0px -40px 0px'
    });

    revealTargets.forEach((el) => observer.observe(el));
  }

  /* Smooth scroll for anchor links */
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const id = link.getAttribute('href');
      if(id && id.length > 1) {
        const target = document.querySelector(id);
        if(target) {
          e.preventDefault();
          target.scrollIntoView({
            behavior: prefersReduced ? 'auto' : 'smooth',
            block: 'start'
          });
        }
      }
    });
  });
})();
