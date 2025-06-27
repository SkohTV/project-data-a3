let currentPage = "home";
let globe = null;
let globeRenderer = null;
let globeScene = null;
let globeCamera = null;
let animationId = null;






document.addEventListener("DOMContentLoaded", function () { // au load du DOM
  setupNavigation();
  setupGlobe();

  setupParallaxEffect();
});





function setupGlobe() { // setup du globe avec le module TREE.js
  const canvas = document.getElementById("globe-canvas");
  if (!canvas) return;

  const width = 1000;
  const height = 1000;

  globeScene = new THREE.Scene();
  globeCamera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
  globeRenderer = new THREE.WebGLRenderer({
    canvas: canvas,
    alpha: true,
    antialias: true,
  });
  globeRenderer.setSize(width, height);
  globeRenderer.setClearColor(0x000000, 0);

  const geometry = new THREE.SphereGeometry(1.2, 64, 64);

  // texture terre
  const canvas2d = document.createElement("canvas");
  canvas2d.width = 1024;
  canvas2d.height = 512;
  const ctx = canvas2d.getContext("2d");

  // ocean
  const gradient = ctx.createRadialGradient(512, 256, 0, 512, 256, 400);
  gradient.addColorStop(0, "#1e40af");
  gradient.addColorStop(0.3, "#0891b2");
  gradient.addColorStop(0.6, "#0369a1");
  gradient.addColorStop(1, "#1e3a8a");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 1024, 512);

  // continents vert
  ctx.fillStyle = "#047857";

  // Amérique du Nord
  ctx.beginPath();
  ctx.ellipse(200, 180, 80, 60, 0, 0, 2 * Math.PI);
  ctx.fill();

  // Amérique du Sud
  ctx.beginPath();
  ctx.ellipse(250, 320, 40, 80, 0, 0, 2 * Math.PI);
  ctx.fill();

  // Afrique
  ctx.beginPath();
  ctx.ellipse(512, 280, 50, 90, 0, 0, 2 * Math.PI);
  ctx.fill();

  // Europe
  ctx.beginPath();
  ctx.ellipse(520, 180, 30, 40, 0, 0, 2 * Math.PI);
  ctx.fill();

  // Asie
  ctx.beginPath();
  ctx.ellipse(700, 200, 100, 70, 0, 0, 2 * Math.PI);
  ctx.fill();

  const texture = new THREE.CanvasTexture(canvas2d);
  const material = new THREE.MeshPhongMaterial({
    map: texture,
    shininess: 5,
    transparent: true,
    opacity: 0.9,
  });

  globe = new THREE.Mesh(geometry, material);
  globeScene.add(globe);


  const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
  globeScene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 3, 5);
  globeScene.add(directionalLight);

  globeCamera.position.z = 3.2;

  animateGlobe(); // l'animation c'est la rotation du globe sur lui meme
}

function animateGlobe() { // rotation du globe
  if (!globe || !globeRenderer || !globeScene || !globeCamera) return;

  animationId = requestAnimationFrame(animateGlobe);
  globe.rotation.y += 0.003;
  globeRenderer.render(globeScene, globeCamera);
}









function setupNavigation() { // navigation entre les pages avec le data-page des balises <a>
  const navLinks = document.querySelectorAll(".nav-link");
  const ctaButton = document.querySelector(".cta-button");
  const featureCards = document.querySelectorAll(".feature-card");

  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetPage = link.getAttribute("data-page");
      navigateToPage(targetPage);
    });
  });

  if (ctaButton) {
    ctaButton.addEventListener("click", (e) => {
      e.preventDefault();
      const targetPage = ctaButton.getAttribute("data-page");
      navigateToPage(targetPage);
    });
  }

  featureCards.forEach((card) => {
    card.addEventListener("click", () => {
      const targetPage = card.getAttribute("data-page");
      if (targetPage) {
        navigateToPage(targetPage);
      }
    });
  });
}



// active or deactivate the page
function navigateToPage(pageId) {
  if (pageId === "home") {
    window.scrollTo(0, 0);
  }

  document.querySelectorAll(".content-page").forEach((page) => {
    page.classList.remove("active");
  });

  const targetPage = document.getElementById(pageId + "-page");
  if (targetPage) {
    targetPage.classList.add("active");
    currentPage = pageId;

    document.querySelectorAll(".nav-link").forEach((link) => {
      link.classList.remove("active");
      if (link.getAttribute("data-page") === pageId) {
        link.classList.add("active");
      }
    });
  }


  if (pageId !== "home" && animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  } else if (pageId === "home" && !animationId) {
    animateGlobe();
  }

  if (pageId == 'predict') {

      // check si il y a une ligne de selectionné
      const selectedRow = document.querySelector('#vessels-table input[type="radio"]:checked');
      if (!selectedRow) {
        document.getElementById("errors").innerHTML = '<strong>Veuillez sélectionner un navire pour prédire sa trajectoire.</strong>';
        document.getElementById("errors").style.display = "block";
        return;
      } else {
        document.getElementById("errors").style.display = "none";
        predict_trajectoire_vesseltype()
      }
      
      
  }
  if (pageId == 'clusters') {
      predict_clusters()
  }
}



function setupParallaxEffect() { //setup de l'effet zoom au scroll
  if (currentPage !== "home") return;

  const earthContainer = document.querySelector(".earth-container");
  const welcomeContent = document.querySelector(".welcome-content");
  const starsLayer = document.querySelector(".stars-layer");

  if (!earthContainer || !welcomeContent) return;

  let ticking = false;

  function updateParallax() {
    if (currentPage !== "home") return;

    const scrollY = window.pageYOffset;
    const windowHeight = window.innerHeight;
    const scrollPercent = Math.min(scrollY / windowHeight, 1);

    if (scrollPercent < 1) {
      const scale = 1 + scrollPercent * 2;
      const translateY = scrollPercent * -100;
      const opacity = 1 - scrollPercent * 1.2;

      if (earthContainer) {
        earthContainer.style.transform = `scale(${scale}) translateY(${translateY}px)`;
      }

      if (welcomeContent) {
        welcomeContent.style.opacity = Math.max(0, opacity);
        welcomeContent.style.transform = `translateY(${translateY * 0.5}px)`;
      }

      if (starsLayer) {
        starsLayer.style.opacity = Math.max(0, 1 - scrollPercent);
      }
    }

    const parallaxScene = document.querySelector(".parallax-scene");
    if (parallaxScene) {
      if (scrollY >= windowHeight) {
        parallaxScene.style.opacity = "0";
        parallaxScene.style.pointerEvents = "none";
      } else {
        parallaxScene.style.opacity = "1";
        parallaxScene.style.pointerEvents = "all";
      }
    }

    ticking = false;
  }

  function requestParallaxUpdate() {
    if (!ticking && currentPage === "home") {
      requestAnimationFrame(updateParallax);
      ticking = true;
    }
  }

  window.addEventListener("scroll", requestParallaxUpdate, { passive: true });
  updateParallax();
}
