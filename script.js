window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function () {
  var options = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 8000,
    initialSlide: 0
  };

  if (window.bulmaCarousel && typeof bulmaCarousel.attach === 'function') {
    bulmaCarousel.attach('.carousel', options);
  }

  if (window.bulmaSlider && typeof bulmaSlider.attach === 'function') {
    bulmaSlider.attach();
  }

  var jumpLinks = Array.prototype.slice.call(
    document.querySelectorAll('.side-jump a, .mobile-jump a')
  );
  var jumpSectionIds = ['top', 'gecko', 'gats', 'experiments', 'discussion', 'conclusion', 'citation'];
  var jumpSections = jumpSectionIds
    .map(function (id) { return document.getElementById(id); })
    .filter(Boolean);

  function setActiveJump(id) {
    jumpLinks.forEach(function (link) {
      link.classList.toggle('is-active', link.getAttribute('href') === ('#' + id));
    });
  }

  if (jumpLinks.length > 0) {
    setActiveJump('top');
    jumpLinks.forEach(function (link) {
      link.addEventListener('click', function () {
        var targetId = (link.getAttribute('href') || '').replace('#', '');
        if (targetId) {
          setActiveJump(targetId);
        }
      });
    });
  }

  if (jumpSections.length > 0 && window.IntersectionObserver) {
    var sectionObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          setActiveJump(entry.target.id);
        }
      });
    }, {
      root: null,
      rootMargin: '-42% 0px -50% 0px',
      threshold: 0
    });

    jumpSections.forEach(function (section) {
      sectionObserver.observe(section);
    });
  }

  var tableScrollWrappers = Array.prototype.slice.call(document.querySelectorAll('.table-scroll'));
  tableScrollWrappers.forEach(function (wrapper) {
    var isDown = false;
    var startX = 0;
    var startScrollLeft = 0;

    wrapper.addEventListener('mousedown', function (event) {
      isDown = true;
      wrapper.classList.add('is-dragging');
      startX = event.pageX;
      startScrollLeft = wrapper.scrollLeft;
      event.preventDefault();
    });

    window.addEventListener('mouseup', function () {
      if (!isDown) {
        return;
      }
      isDown = false;
      wrapper.classList.remove('is-dragging');
    });

    wrapper.addEventListener('mouseleave', function () {
      if (!isDown) {
        return;
      }
      isDown = false;
      wrapper.classList.remove('is-dragging');
    });

    wrapper.addEventListener('mousemove', function (event) {
      if (!isDown) {
        return;
      }
      var delta = (event.pageX - startX) * 1.1;
      wrapper.scrollLeft = startScrollLeft - delta;
      event.preventDefault();
    });
  });

  var viewer = document.getElementById('image-viewer');
  var viewerImage = document.getElementById('viewer-image');
  if (!viewer || !viewerImage) {
    return;
  }
  var zoomableImages = Array.prototype.slice.call(document.querySelectorAll('img'))
    .filter(function (img) {
      return img.id !== 'viewer-image' && !!img.src;
    });
  if (zoomableImages.length === 0) {
    return;
  }

  var scale = 1;
  var translateX = 0;
  var translateY = 0;
  var minScale = 1;
  var maxScale = 8;
  var isDragging = false;
  var pointerX = 0;
  var pointerY = 0;
  var pinchStartDistance = 0;
  var pinchStartScale = 1;
  var touchMode = '';
  var lastInteractionAt = 0;

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function isViewerOpen() {
    return viewer.classList.contains('is-open');
  }

  function updateCursor() {
    if (!isViewerOpen()) {
      return;
    }
    if (isDragging) {
      viewerImage.style.cursor = 'grabbing';
    } else if (scale > 1) {
      viewerImage.style.cursor = 'grab';
    } else {
      viewerImage.style.cursor = 'zoom-out';
    }
  }

  function updateTransform() {
    viewerImage.style.transform =
      'translate(' + translateX + 'px, ' + translateY + 'px) scale(' + scale + ')';
    updateCursor();
  }

  function resetTransform() {
    scale = 1;
    translateX = 0;
    translateY = 0;
    isDragging = false;
    updateTransform();
  }

  function openViewer(src, alt) {
    viewerImage.src = src;
    viewerImage.alt = alt || 'Expanded figure';
    viewer.classList.add('is-open');
    viewer.setAttribute('aria-hidden', 'false');
    document.body.classList.add('viewer-open');
    resetTransform();
  }

  function closeViewer() {
    viewer.classList.remove('is-open');
    viewer.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('viewer-open');
  }

  function zoomAt(nextScale, centerX, centerY) {
    var targetScale = clamp(nextScale, minScale, maxScale);
    if (targetScale === scale) {
      return;
    }

    var rect = viewer.getBoundingClientRect();
    var offsetX = centerX - (rect.left + rect.width / 2) - translateX;
    var offsetY = centerY - (rect.top + rect.height / 2) - translateY;
    var ratio = targetScale / scale;

    translateX -= offsetX * (ratio - 1);
    translateY -= offsetY * (ratio - 1);
    scale = targetScale;
    updateTransform();
  }

  function touchDistance(touchA, touchB) {
    var dx = touchA.clientX - touchB.clientX;
    var dy = touchA.clientY - touchB.clientY;
    return Math.hypot(dx, dy);
  }

  function touchCenter(touchA, touchB) {
    return {
      x: (touchA.clientX + touchB.clientX) / 2,
      y: (touchA.clientY + touchB.clientY) / 2
    };
  }

  zoomableImages.forEach(function (img) {
    img.classList.add('zoomable-image');
    img.addEventListener('click', function (event) {
      event.preventDefault();
      event.stopPropagation();
      openViewer(img.currentSrc || img.src, img.alt);
      lastInteractionAt = Date.now();
    });
  });

  viewer.addEventListener('wheel', function (event) {
    if (!isViewerOpen()) {
      return;
    }
    event.preventDefault();
    var zoomFactor = event.deltaY < 0 ? 1.14 : 0.88;
    zoomAt(scale * zoomFactor, event.clientX, event.clientY);
    lastInteractionAt = Date.now();
  }, { passive: false });

  viewerImage.addEventListener('mousedown', function (event) {
    if (!isViewerOpen() || scale <= 1) {
      return;
    }
    isDragging = true;
    pointerX = event.clientX;
    pointerY = event.clientY;
    updateCursor();
    event.preventDefault();
  });

  window.addEventListener('mousemove', function (event) {
    if (!isDragging || !isViewerOpen()) {
      return;
    }
    translateX += event.clientX - pointerX;
    translateY += event.clientY - pointerY;
    pointerX = event.clientX;
    pointerY = event.clientY;
    updateTransform();
    lastInteractionAt = Date.now();
  });

  window.addEventListener('mouseup', function () {
    if (!isDragging) {
      return;
    }
    isDragging = false;
    updateCursor();
    lastInteractionAt = Date.now();
  });

  viewer.addEventListener('touchstart', function (event) {
    if (!isViewerOpen()) {
      return;
    }

    if (event.touches.length === 2) {
      touchMode = 'pinch';
      pinchStartDistance = touchDistance(event.touches[0], event.touches[1]);
      pinchStartScale = scale;
      lastInteractionAt = Date.now();
    } else if (event.touches.length === 1) {
      touchMode = 'drag';
      pointerX = event.touches[0].clientX;
      pointerY = event.touches[0].clientY;
    }
  }, { passive: false });

  viewer.addEventListener('touchmove', function (event) {
    if (!isViewerOpen()) {
      return;
    }

    if (touchMode === 'pinch' && event.touches.length === 2) {
      event.preventDefault();
      var currentDistance = touchDistance(event.touches[0], event.touches[1]);
      if (pinchStartDistance > 0) {
        var center = touchCenter(event.touches[0], event.touches[1]);
        zoomAt(pinchStartScale * (currentDistance / pinchStartDistance), center.x, center.y);
      }
      lastInteractionAt = Date.now();
      return;
    }

    if (touchMode === 'drag' && event.touches.length === 1 && scale > 1) {
      event.preventDefault();
      var touch = event.touches[0];
      translateX += touch.clientX - pointerX;
      translateY += touch.clientY - pointerY;
      pointerX = touch.clientX;
      pointerY = touch.clientY;
      updateTransform();
      lastInteractionAt = Date.now();
    }
  }, { passive: false });

  viewer.addEventListener('touchend', function (event) {
    if (!isViewerOpen()) {
      return;
    }

    if (event.touches.length === 2) {
      touchMode = 'pinch';
      pinchStartDistance = touchDistance(event.touches[0], event.touches[1]);
      pinchStartScale = scale;
      return;
    }

    if (event.touches.length === 1) {
      touchMode = 'drag';
      pointerX = event.touches[0].clientX;
      pointerY = event.touches[0].clientY;
      return;
    }

    touchMode = '';
    pinchStartDistance = 0;
  });

  viewer.addEventListener('click', function () {
    if (!isViewerOpen()) {
      return;
    }
    if (Date.now() - lastInteractionAt < 220) {
      return;
    }
    closeViewer();
  });

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && isViewerOpen()) {
      closeViewer();
    }
  });
});
