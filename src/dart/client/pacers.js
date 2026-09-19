/**
 * DART consumer pacers. The console is a demo; this file is the browser SDK.
 *
 * Outstanding Interests are the only thing that may run a decode kernel.
 * Each pacer's nextWindow() is W — credit the GPU can spend, nothing else.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.DartPacers = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  class DrainPacer {
    constructor(window) {
      this.name = "drain";
      this.window = window || 32;
    }
    nextWindow() {
      return Promise.resolve(this.window);
    }
  }

  class ReadingPacer {
    constructor(opts) {
      opts = opts || {};
      this.name = "reading";
      this.tokensPerSec = opts.tokensPerSec || 30;
      this.burst = opts.burst || 16;
      this._tokens = this.burst;
      this._last = (typeof performance !== "undefined" ? performance.now() : Date.now()) / 1000;
    }
    nextWindow() {
      const self = this;
      return new Promise(function poll(resolve) {
        const now = (typeof performance !== "undefined" ? performance.now() : Date.now()) / 1000;
        const elapsed = now - self._last;
        self._last = now;
        self._tokens = Math.min(self.burst * 2, self._tokens + elapsed * self.tokensPerSec);
        if (self._tokens >= 1) {
          const n = Math.min(self.burst, Math.floor(self._tokens));
          self._tokens -= n;
          resolve(Math.max(1, n));
          return;
        }
        const need = (1 - self._tokens) / self.tokensPerSec;
        setTimeout(function () {
          poll(resolve);
        }, Math.max(8, need * 1000));
      });
    }
  }

  class TtsPacer {
    constructor(opts) {
      opts = opts || {};
      this.name = "tts";
      this.realtimeFactor = opts.realtimeFactor == null ? 1 : opts.realtimeFactor;
      this._inner = new ReadingPacer({
        tokensPerSec: (opts.tokensPerSec || 18) * this.realtimeFactor,
        burst: opts.burst || 12,
      });
    }
    nextWindow() {
      return this._inner.nextWindow();
    }
  }

  class JsonNeedPacer {
    constructor(opts) {
      opts = opts || {};
      this.name = "json";
      this.burst = opts.burst || 64;
      this.pauseMs = (opts.pauseS || 0.5) * 1000;
      this._paused = false;
    }
    nextWindow() {
      const self = this;
      if (!self._paused) {
        self._paused = true;
        return Promise.resolve(self.burst);
      }
      return new Promise(function (resolve) {
        setTimeout(function () {
          self._paused = true;
          resolve(self.burst);
        }, self.pauseMs);
      });
    }
  }

  class ToolCallPacer {
    constructor(opts) {
      opts = opts || {};
      this.name = "tool";
      this.burst = opts.burst || 64;
      this._ack = null;
      this._armed = true;
    }
    ack() {
      this._armed = true;
      if (this._ack) {
        const fn = this._ack;
        this._ack = null;
        fn();
      }
    }
    nextWindow() {
      const self = this;
      if (self._armed) {
        self._armed = false;
        return Promise.resolve(self.burst);
      }
      return new Promise(function (resolve) {
        self._ack = function () {
          self._armed = false;
          resolve(self.burst);
        };
      });
    }
  }

  class ViewportPacer {
    /**
     * Credit only while `el` intersects the viewport (IntersectionObserver).
     */
    constructor(el, opts) {
      opts = opts || {};
      this.name = "viewport";
      this.burst = opts.burst || 16;
      this._visible = false;
      this._waiters = [];
      const self = this;
      this._io = new IntersectionObserver(
        function (entries) {
          self.observe(entries.some(function (e) {
            return e.isIntersecting;
          }));
        },
        { threshold: opts.threshold == null ? 0.25 : opts.threshold }
      );
      if (el) {
        this._io.observe(el);
      }
    }
    observe(visible) {
      this._visible = !!visible;
      if (this._visible && this._waiters.length) {
        const burst = this.burst;
        this._waiters.splice(0).forEach(function (fn) {
          fn(burst);
        });
      }
    }
    unobserve() {
      this.observe(false);
      if (this._io) {
        this._io.disconnect();
      }
    }
    nextWindow() {
      if (this._visible) {
        return Promise.resolve(this.burst);
      }
      const self = this;
      return new Promise(function (resolve) {
        self._waiters.push(resolve);
      });
    }
  }

  return {
    DrainPacer: DrainPacer,
    ReadingPacer: ReadingPacer,
    TtsPacer: TtsPacer,
    JsonNeedPacer: JsonNeedPacer,
    ToolCallPacer: ToolCallPacer,
    ViewportPacer: ViewportPacer,
  };
});
