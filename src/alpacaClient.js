const PAPER_TRADING_BASE_URL = 'https://paper-api.alpaca.markets';
const LIVE_TRADING_BASE_URL = 'https://api.alpaca.markets';
const MARKET_DATA_BASE_URL = 'https://data.alpaca.markets';

export class AlpacaApiError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = 'AlpacaApiError';
    this.status = details.status;
    this.statusText = details.statusText;
    this.body = details.body;
    this.requestId = details.requestId;
  }
}

export class AlpacaClient {
  constructor(config) {
    this.config = config;
  }

  async getAccount() {
    return this.request('/v2/account');
  }

  async getClock() {
    return this.request('/v2/clock');
  }

  async getAsset(symbol) {
    return this.request(`/v2/assets/${encodeURIComponent(symbol.toUpperCase())}`);
  }

  async listPositions() {
    return this.request('/v2/positions');
  }

  async listOrders(params = {}) {
    return this.request('/v2/orders', {
      query: {
        status: params.status ?? 'open',
        limit: params.limit ?? 50,
        direction: params.direction ?? 'desc',
        after: params.after,
        until: params.until,
        symbols: Array.isArray(params.symbols) ? params.symbols.join(',') : params.symbols
      }
    });
  }

  async submitOrder(order) {
    return this.request('/v2/orders', {
      method: 'POST',
      body: order
    });
  }

  async cancelOrder(orderId) {
    return this.request(`/v2/orders/${encodeURIComponent(orderId)}`, {
      method: 'DELETE'
    });
  }

  async getLatestStockQuote(symbol) {
    return this.request(`/v2/stocks/${encodeURIComponent(symbol.toUpperCase())}/quotes/latest`, {
      baseUrl: this.config.dataBaseUrl
    });
  }

  async request(endpoint, options = {}) {
    ensureFetchExists();

    const method = options.method ?? 'GET';
    const baseUrl = options.baseUrl ?? this.config.baseUrl;

    assertSafeTradingRequest({ baseUrl, method, allowLiveTrading: this.config.allowLiveTrading });

    const url = new URL(endpoint, `${baseUrl}/`);
    appendQuery(url, options.query);

    const response = await fetch(url, {
      method,
      headers: buildHeaders(this.config, options.body),
      body: options.body === undefined ? undefined : JSON.stringify(options.body)
    });

    const body = await readResponseBody(response);

    if (!response.ok) {
      throw new AlpacaApiError(formatApiError(response, body), {
        status: response.status,
        statusText: response.statusText,
        body,
        requestId: response.headers.get('x-request-id')
      });
    }

    return body;
  }
}

export function createAlpacaClient(overrides = {}) {
  return new AlpacaClient(readAlpacaConfig(overrides));
}

export function readAlpacaConfig(overrides = {}) {
  const keyId = overrides.keyId ?? process.env.APCA_API_KEY_ID ?? process.env.ALPACA_API_KEY_ID;
  const secretKey =
    overrides.secretKey ?? process.env.APCA_API_SECRET_KEY ?? process.env.ALPACA_API_SECRET_KEY;

  if (!keyId || !secretKey) {
    throw new Error(
      'Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env or your shell.'
    );
  }

  return {
    keyId,
    secretKey,
    baseUrl: normalizeBaseUrl(overrides.baseUrl ?? process.env.APCA_API_BASE_URL ?? PAPER_TRADING_BASE_URL),
    dataBaseUrl: normalizeBaseUrl(overrides.dataBaseUrl ?? process.env.APCA_DATA_BASE_URL ?? MARKET_DATA_BASE_URL),
    allowLiveTrading:
      overrides.allowLiveTrading ??
      ['1', 'true', 'yes'].includes(String(process.env.ALPACA_ALLOW_LIVE_TRADING).toLowerCase())
  };
}

export function isLiveTradingBaseUrl(baseUrl) {
  return normalizeBaseUrl(baseUrl) === LIVE_TRADING_BASE_URL;
}

function buildHeaders(config, body) {
  const headers = {
    Accept: 'application/json',
    'APCA-API-KEY-ID': config.keyId,
    'APCA-API-SECRET-KEY': config.secretKey
  };

  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
  }

  return headers;
}

function appendQuery(url, query = {}) {
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(key, value);
    }
  }
}

async function readResponseBody(response) {
  if (response.status === 204) {
    return null;
  }

  const text = await response.text();

  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function formatApiError(response, body) {
  const message = typeof body === 'object' && body !== null ? body.message : body;
  return `Alpaca API request failed (${response.status} ${response.statusText})${message ? `: ${message}` : ''}`;
}

function assertSafeTradingRequest({ baseUrl, method, allowLiveTrading }) {
  const isWrite = !['GET', 'HEAD', 'OPTIONS'].includes(method.toUpperCase());

  if (isWrite && isLiveTradingBaseUrl(baseUrl) && !allowLiveTrading) {
    throw new Error(
      'Refusing to send a write request to live Alpaca trading. Set ALPACA_ALLOW_LIVE_TRADING=true to opt in.'
    );
  }
}

function ensureFetchExists() {
  if (typeof fetch !== 'function') {
    throw new Error('This Alpaca integration requires Node.js 18 or newer for the built-in fetch API.');
  }
}

function normalizeBaseUrl(value) {
  return String(value).replace(/\/+$/, '');
}
