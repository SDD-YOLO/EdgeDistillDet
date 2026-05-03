const requestInterceptors = new Set();
const responseInterceptors = new Set();
const errorInterceptors = new Set();

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function createAbortController(signal, timeoutMs) {
  const controller = new AbortController();
  const cleanups = [];

  if (signal) {
    if (signal.aborted) {
      controller.abort(signal.reason);
    } else {
      const onAbort = () => controller.abort(signal.reason);
      signal.addEventListener("abort", onAbort, { once: true });
      cleanups.push(() => signal.removeEventListener("abort", onAbort));
    }
  }

  if (timeoutMs > 0) {
    const timeoutId = window.setTimeout(
      () => controller.abort(new DOMException("请求超时", "AbortError")),
      timeoutMs,
    );
    cleanups.push(() => window.clearTimeout(timeoutId));
  }

  return {
    controller,
    cleanup() {
      cleanups.forEach((fn) => fn());
    },
  };
}

function isRetryableStatus(status) {
  return (
    status === 408 ||
    status === 425 ||
    status === 429 ||
    (status >= 500 && status < 600)
  );
}

export function addApiRequestInterceptor(interceptor) {
  requestInterceptors.add(interceptor);
  return () => requestInterceptors.delete(interceptor);
}

export function addApiResponseInterceptor(interceptor) {
  responseInterceptors.add(interceptor);
  return () => responseInterceptors.delete(interceptor);
}

export function addApiErrorInterceptor(interceptor) {
  errorInterceptors.add(interceptor);
  return () => errorInterceptors.delete(interceptor);
}

export async function apiRequest(url, options = {}) {
  const {
    headers,
    signal,
    timeoutMs = 0,
    retries = 0,
    retryDelayMs = 250,
    ...fetchOptions
  } = options;
  const requestState = {
    url,
    options: {
      ...fetchOptions,
      headers: { "Content-Type": "application/json", ...headers },
      signal,
    },
  };

  for (const interceptor of requestInterceptors) {
    // Interceptors can mutate or replace the request descriptor.
    const nextRequest = await interceptor(requestState);
    if (nextRequest && typeof nextRequest === "object") {
      requestState.url = nextRequest.url || requestState.url;
      requestState.options = {
        ...requestState.options,
        ...nextRequest.options,
      };
    }
  }

  let lastError = null;

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const { controller, cleanup } = createAbortController(
      requestState.options.signal,
      timeoutMs,
    );
    try {
      const response = await fetch(requestState.url, {
        ...requestState.options,
        signal: controller.signal,
      });

      const data = await response.json();
      if (!response.ok) {
        const err = new Error(data.error || "请求失败");
        err.status = response.status;
        err.payload = data;
        throw err;
      }

      let nextData = data;
      for (const interceptor of responseInterceptors) {
        const intercepted = await interceptor(nextData, {
          url: requestState.url,
          options: requestState.options,
          response,
        });
        if (intercepted !== undefined) nextData = intercepted;
      }
      return nextData;
    } catch (error) {
      lastError = error;
      for (const interceptor of errorInterceptors) {
        const intercepted = await interceptor(error, {
          url: requestState.url,
          options: requestState.options,
          attempt,
        });
        if (intercepted !== undefined) lastError = intercepted;
      }

      const canRetry =
        attempt < retries &&
        error?.name !== "AbortError" &&
        (error?.status == null || isRetryableStatus(error.status));
      if (!canRetry) {
        throw lastError;
      }

      await sleep(retryDelayMs * (attempt + 1));
    } finally {
      cleanup();
    }
  }

  throw lastError || new Error("请求失败");
}
