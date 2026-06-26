window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

var COMPARISON_BASE = "static/videos/comparison";
var COMPARISON_SCENES = [
  {
    label: "1",
    index: "00030",
    title: "Scene 30",
    slug: "cochem_kitchen_view_0_images_0"
  },
  {
    label: "2",
    index: "00080",
    title: "Scene 80",
    slug: "fingerhut_childrens_room_view_0_images_0"
  },
  {
    label: "3",
    index: "00115",
    title: "Scene 115",
    slug: "fingerhut_living_room_view_2_images_0"
  },
  {
    label: "4",
    index: "00140",
    title: "Scene 140",
    slug: "fingerhut_living_room_view_7_images_0"
  },
  {
    label: "5",
    index: "00150",
    title: "Scene 150",
    slug: "fingerhut_upstairs_bathroom_1_view_0_images_0"
  }
];
var COMPARISON_TURNS = [
  { key: "turn_on", label: "Turn On" },
  { key: "turn_off", label: "Turn Off" }
];
var COMPARISON_VIEWS = [
  { key: "rgb", label: "RGB" },
  { key: "sie", label: "SIE Heatmap", metricFile: "standardized_ratio_diff.jpg" },
  { key: "lfe", label: "LFE Heatmap", metricFile: "gradient_diff.jpg" }
];
var COMPARISON_MODELS = [
  { folder: "flux_2_dev", label: "FLUX.2 Dev", metricIndex: "03" },
  { folder: "flux_2_max", label: "FLUX.2 Max", metricIndex: "04" },
  { folder: "gpt_image_1.5", label: "GPT-Image 1.5", metricIndex: "01" },
  { folder: "nano_banana_2", label: "Nano Banana 2", metricIndex: "05" },
  { folder: "nano_banana_pro", label: "Nano Banana Pro", metricIndex: "06" },
  { folder: "qwen_image_edit_2511", label: "Qwen Image Edit 2511", metricIndex: "08" }
];
var COMPARISON_METRICS = {};

function comparisonFolder(scene, turnKey) {
  return COMPARISON_BASE + "/" + scene.index + "_" + scene.slug;
}

function comparisonTargetState(turnKey) {
  return turnKey === "turn_on" ? "on" : "off";
}

function comparisonInputState(turnKey) {
  return turnKey === "turn_on" ? "off" : "on";
}

function comparisonGroundTruthPath(scene, state) {
  var sceneId = scene.index + "_" + scene.slug;
  return comparisonFolder(scene) + "/ground_truth/image_only/" + sceneId + "_gt_" + state + ".jpg";
}

function comparisonRgbPath(scene, turnKey, model) {
  var sceneId = scene.index + "_" + scene.slug;
  return comparisonFolder(scene) + "/ai_generated/image_only/" + model.folder + "/" +
    sceneId + "_" + model.folder + "_" + comparisonTargetState(turnKey) + ".jpg";
}

function comparisonMetricPath(scene, turnKey, model, view) {
  var sceneId = scene.index + "_" + scene.slug;
  var metricStep = view.key === "sie" ? "02" : "04";
  return comparisonFolder(scene) + "/metrics/" + turnKey + "/parts/image_only/models/" + model.folder + "/" +
    sceneId + "_" + turnKey + "_" + model.metricIndex + "_" + metricStep + "_" + view.metricFile;
}

function comparisonImagePath(scene, turnKey, model, view) {
  return view.key === "rgb"
    ? comparisonRgbPath(scene, turnKey, model)
    : comparisonMetricPath(scene, turnKey, model, view);
}

function formatComparisonMetric(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "-";
}

function createComparisonTab(label, isActive, onClick) {
  var item = document.createElement("li");
  var button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.setAttribute("aria-pressed", isActive ? "true" : "false");
  button.addEventListener("click", onClick);
  if (isActive) item.className = "is-active";
  item.appendChild(button);
  return item;
}

function createComparisonFigure(src, caption, alt, metrics) {
  var figure = document.createElement("figure");
  figure.className = "comparison-figure";

  var image = document.createElement("img");
  image.src = src;
  image.alt = alt;
  image.loading = "lazy";
  image.decoding = "async";

  var figcaption = document.createElement("figcaption");
  figcaption.textContent = caption;

  figure.appendChild(image);
  figure.appendChild(figcaption);

  if (metrics) {
    var metricLine = document.createElement("div");
    metricLine.className = "comparison-metrics";
    metricLine.textContent = "SIE: " + formatComparisonMetric(metrics.sie) + " | LFE: " + formatComparisonMetric(metrics.lfe);
    figure.appendChild(metricLine);
  }

  return figure;
}

function initComparisonGallery() {
  var viewer = document.getElementById("comparison-viewer");
  if (!viewer) return;

  var sceneTabs = document.getElementById("comparison-scene-tabs");
  var turnTabs = document.getElementById("comparison-turn-tabs");
  var viewTabs = document.getElementById("comparison-view-tabs");
  var referenceGrid = document.getElementById("comparison-reference-grid");
  var modelGrid = document.getElementById("comparison-model-grid");
  var activeSceneIndex = 0;
  var activeTurnKey = "turn_on";
  var activeViewKey = "rgb";

  function getActiveView() {
    return COMPARISON_VIEWS.find(function(view) {
      return view.key === activeViewKey;
    }) || COMPARISON_VIEWS[0];
  }

  function getComparisonMetrics(scene, turnKey, model) {
    return (((COMPARISON_METRICS[scene.index] || {})[turnKey] || {})[model.folder]);
  }

  function renderTabs() {
    sceneTabs.replaceChildren();
    COMPARISON_SCENES.forEach(function(scene, index) {
      sceneTabs.appendChild(createComparisonTab(scene.label, index === activeSceneIndex, function() {
        activeSceneIndex = index;
        render();
      }));
    });

    turnTabs.replaceChildren();
    COMPARISON_TURNS.forEach(function(turn) {
      turnTabs.appendChild(createComparisonTab(turn.label, turn.key === activeTurnKey, function() {
        activeTurnKey = turn.key;
        render();
      }));
    });

    viewTabs.replaceChildren();
    COMPARISON_VIEWS.forEach(function(view) {
      viewTabs.appendChild(createComparisonTab(view.label, view.key === activeViewKey, function() {
        activeViewKey = view.key;
        render();
      }));
    });
  }

  function renderImages() {
    var scene = COMPARISON_SCENES[activeSceneIndex];
    var activeView = getActiveView();

    referenceGrid.replaceChildren(
      createComparisonFigure(
        comparisonGroundTruthPath(scene, comparisonInputState(activeTurnKey)),
        "Input",
        scene.title + " input image"
      ),
      createComparisonFigure(
        comparisonGroundTruthPath(scene, comparisonTargetState(activeTurnKey)),
        "GT Output",
        scene.title + " ground-truth output image"
      )
    );

    modelGrid.replaceChildren();
    COMPARISON_MODELS.forEach(function(model) {
      modelGrid.appendChild(createComparisonFigure(
        comparisonImagePath(scene, activeTurnKey, model, activeView),
        model.label,
        scene.title + " " + model.label + " " + activeView.label.toLowerCase(),
        getComparisonMetrics(scene, activeTurnKey, model)
      ));
    });
  }

  function render() {
    renderTabs();
    renderImages();
  }

  render();
}

function initTitleAnswerStrip() {
  var strip = document.getElementById("title-answer-strip");
  if (!strip) return;

  strip.querySelectorAll("img[data-sie-src]").forEach(function(image) {
    image.setAttribute("data-rgb-src", image.src);
  });

  function setRevealed(isRevealed) {
    strip.querySelectorAll("img[data-sie-src]").forEach(function(image) {
      image.src = image.getAttribute(isRevealed ? "data-sie-src" : "data-rgb-src");
      image.alt = image.alt
        .replace(isRevealed ? "RGB prediction" : "SIE heatmap", isRevealed ? "SIE heatmap" : "RGB prediction");
    });
    strip.querySelectorAll(".title-answer-verdict[data-verdict]").forEach(function(verdict) {
      verdict.textContent = isRevealed ? verdict.getAttribute("data-verdict") : "Maybe?";
    });
    strip.classList.toggle("is-revealed", isRevealed);
  }

  function toggleReveal() {
    setRevealed(!strip.classList.contains("is-revealed"));
  }

  strip.addEventListener("click", toggleReveal);
  strip.addEventListener("keydown", function(event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggleReveal();
    }
  });
}


function initBeforeAfterSlider(containerId) {
  var container = document.getElementById(containerId);
  if (!container) return;

  var before = container.querySelector('.bal-before');
  var beforeInset = container.querySelector('.bal-before-inset');
  var handle = container.querySelector('.bal-handle');
  var afterMedia = container.querySelector('.bal-after video, .bal-after img');
  var beforeVideo = container.querySelector('.bal-before-inset video');
  var afterVideo = container.querySelector('.bal-after video');

  var isDragging = false;

  function setPosition(pct) {
    pct = Math.max(0, Math.min(100, pct));
    before.style.width = pct + '%';
    beforeInset.style.width = container.offsetWidth + 'px';
    handle.style.left = pct + '%';
  }

  function getClientX(e) {
    return e.touches ? e.touches[0].clientX : e.clientX;
  }

  function onMove(e) {
    if (!isDragging) return;
    var rect = container.getBoundingClientRect();
    setPosition(((getClientX(e) - rect.left) / rect.width) * 100);
  }

  function onUp() { isDragging = false; }
  function onDown(e) { isDragging = true; e.preventDefault(); }

  container.addEventListener('mousedown', onDown);
  container.addEventListener('touchstart', onDown, { passive: false });
  document.addEventListener('mousemove', onMove);
  document.addEventListener('touchmove', onMove, { passive: false });
  document.addEventListener('mouseup', onUp);
  document.addEventListener('touchend', onUp);

  window.addEventListener('resize', function() { setPosition(50); });

  function init() {
    beforeInset.style.width = container.offsetWidth + 'px';
    setPosition(50);
  }

  if (afterMedia && afterMedia.tagName === 'IMG') {
    if (afterMedia.complete) { init(); } else { afterMedia.addEventListener('load', init); }
  } else if (afterMedia) {
    if (afterMedia.readyState >= 1) { init(); } else { afterMedia.addEventListener('loadedmetadata', init); }
  }

  // Keep the two videos in this slider in sync (resync if they drift > 0.1s).
  if (beforeVideo && afterVideo) {
    afterVideo.addEventListener('timeupdate', function() {
      if (Math.abs(beforeVideo.currentTime - afterVideo.currentTime) > 0.1) {
        beforeVideo.currentTime = afterVideo.currentTime;
      }
    });
  }
}

$(document).ready(function() {
    $('.dataset-teaser-frame').each(function() {
      var frame = this;
      var video = frame.querySelector('.dataset-teaser-video');
      var zoomVideo = frame.querySelector('.dataset-teaser-zoom-video');
      var lens = frame.querySelector('.dataset-video-lens');
      var toggle = frame.querySelector('.dataset-video-toggle');
      var playbackRate = 0.5;
      var zoomScale = 3;

      if (!video || !zoomVideo || !lens || !toggle) return;

      function setPlaybackRate() {
        video.playbackRate = playbackRate;
        zoomVideo.playbackRate = playbackRate;
      }

      function syncZoomVideo() {
        if (zoomVideo.readyState < 1) return;
        if (Math.abs(zoomVideo.currentTime - video.currentTime) > 0.05) {
          zoomVideo.currentTime = video.currentTime;
        }
      }

      function playZoomVideo() {
        var playPromise = zoomVideo.play();
        if (playPromise && playPromise.catch) {
          playPromise.catch(function() {});
        }
      }

      function updateToggle() {
        toggle.classList.toggle('is-paused', video.paused);
        toggle.setAttribute('aria-label', video.paused ? 'Play video' : 'Pause video');
        toggle.setAttribute('title', video.paused ? 'Play video' : 'Pause video');
      }

      function updateLens(event) {
        if (event.target.closest && event.target.closest('.dataset-video-toggle')) {
          frame.classList.remove('is-zooming');
          return;
        }

        var rect = frame.getBoundingClientRect();
        var x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
        var y = Math.max(0, Math.min(rect.height, event.clientY - rect.top));
        var lensSize = lens.offsetWidth || 170;

        lens.style.setProperty('--lens-x', x + 'px');
        lens.style.setProperty('--lens-y', y + 'px');
        zoomVideo.style.width = rect.width + 'px';
        zoomVideo.style.height = rect.height + 'px';
        zoomVideo.style.transform = 'translate(' + (lensSize / 2 - x * zoomScale) + 'px, ' + (lensSize / 2 - y * zoomScale) + 'px) scale(' + zoomScale + ')';
        frame.classList.add('is-zooming');
      }

      setPlaybackRate();
      updateToggle();

      video.addEventListener('loadedmetadata', setPlaybackRate);
      zoomVideo.addEventListener('loadedmetadata', function() {
        setPlaybackRate();
        syncZoomVideo();
      });

      video.addEventListener('play', function() {
        updateToggle();
        syncZoomVideo();
        playZoomVideo();
      });

      video.addEventListener('pause', function() {
        updateToggle();
        zoomVideo.pause();
      });

      video.addEventListener('timeupdate', syncZoomVideo);

      toggle.addEventListener('click', function() {
        if (video.paused) {
          var playPromise = video.play();
          if (playPromise && playPromise.catch) {
            playPromise.catch(function() {});
          }
          syncZoomVideo();
          playZoomVideo();
        } else {
          video.pause();
        }
      });

      toggle.addEventListener('pointerleave', function() {
        frame.classList.remove('is-zooming');
      });

      frame.addEventListener('pointerenter', function(event) {
        if (event.pointerType === 'touch') return;
        frame.classList.add('is-zooming');
        syncZoomVideo();
        updateLens(event);
        if (!video.paused) playZoomVideo();
      });

      frame.addEventListener('pointermove', function(event) {
        if (event.pointerType === 'touch') return;
        updateLens(event);
      });

      frame.addEventListener('pointerleave', function() {
        frame.classList.remove('is-zooming');
      });
    });

    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

    initTitleAnswerStrip();
    initComparisonGallery();

})
