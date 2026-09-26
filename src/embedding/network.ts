import { execFileSync } from 'child_process';

/**
 * Finds the way out to the internet on a corporate network.
 *
 * On a managed machine the developer has usually never had to think about the
 * proxy: the browser and npm are configured for them. Node's fetch is not, so a
 * download that works everywhere else fails here with a bare connection reset
 * and nothing to act on. Working it out ourselves turns that into a download
 * that simply succeeds.
 */

/** Environment variables that name a proxy, in the order they should win. */
const PROXY_VARIABLES = ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy'];

/** Where Windows records the browser's proxy configuration. */
const INTERNET_SETTINGS =
  'HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings';

/**
 * The proxy to reach an external host through, or null to connect directly.
 *
 * Explicit configuration is honoured first, because a developer who set it
 * meant it. Only then does the machine's own browser configuration get read.
 */
export function resolveProxy(): string | null {
  for (const name of PROXY_VARIABLES) {
    const value = process.env[name];
    if (value && value.trim().length > 0) return normalize(value.trim());
  }

  if (process.platform === 'win32') {
    return resolveWindowsProxy();
  }

  return null;
}

/**
 * Reads the proxy the browser uses on this machine.
 *
 * Two forms are in use: a fixed proxy, or a script that decides per URL. The
 * script is the common one in large organisations, and the address for external
 * traffic is written plainly enough inside it to be picked out without running
 * any of its JavaScript.
 */
function resolveWindowsProxy(): string | null {
  const settings = readInternetSettings();
  if (!settings) return null;

  if (settings.proxyEnabled && settings.proxyServer) {
    return normalize(proxyForHttps(settings.proxyServer));
  }

  if (settings.autoConfigUrl) {
    return proxyFromScript(settings.autoConfigUrl);
  }

  return null;
}

interface InternetSettings {
  proxyEnabled: boolean;
  proxyServer: string | null;
  autoConfigUrl: string | null;
}

function readInternetSettings(): InternetSettings | null {
  let output: string;
  try {
    output = execFileSync('reg', ['query', INTERNET_SETTINGS], {
      encoding: 'utf-8',
      timeout: 5000,
      stdio: ['ignore', 'pipe', 'ignore'],
    });
  } catch {
    return null;
  }

  return {
    proxyEnabled: /ProxyEnable\s+REG_DWORD\s+0x1/i.test(output),
    proxyServer: valueOf(output, 'ProxyServer'),
    autoConfigUrl: valueOf(output, 'AutoConfigURL'),
  };
}

function valueOf(registryOutput: string, name: string): string | null {
  const match = registryOutput.match(new RegExp(`${name}\\s+REG_[A-Z_]+\\s+(.+)`, 'i'));
  return match ? match[1].trim() : null;
}

/**
 * A fixed proxy setting may name one server, or a different one per protocol
 * as "http=host:port;https=host:port". Secure traffic is what we make.
 */
function proxyForHttps(proxyServer: string): string {
  if (!proxyServer.includes('=')) return proxyServer;

  for (const part of proxyServer.split(';')) {
    const [protocol, address] = part.split('=');
    if (protocol?.trim().toLowerCase() === 'https' && address) return address.trim();
  }

  const first = proxyServer.split(';')[0];
  return first.includes('=') ? first.split('=')[1].trim() : first.trim();
}

/**
 * Extracts the proxy an auto-configuration script sends external traffic to.
 *
 * The script is a function deciding per URL, and running it properly would mean
 * implementing the host-matching helpers it calls. It is not worth that: the
 * fallback it returns for everything not on the internal network is written as
 * a plain "PROXY host:port", and that is the one we want. If the file cannot be
 * read or holds nothing recognisable, we simply report no proxy and the caller
 * tries a direct connection.
 */
function proxyFromScript(scriptUrl: string): string | null {
  let script: string;
  try {
    script = fetchSynchronously(scriptUrl);
  } catch {
    return null;
  }

  // The last PROXY directive is the general fallback; earlier ones tend to be
  // special cases for particular hosts.
  const directives = [...script.matchAll(/PROXY\s+([A-Za-z0-9_.-]+:\d+)/g)];
  if (directives.length === 0) return null;

  return normalize(directives[directives.length - 1][1]);
}

/**
 * Reads a small file over HTTP without async, so proxy discovery can complete
 * before the first request is made.
 */
function fetchSynchronously(url: string): string {
  return execFileSync(
    process.execPath,
    [
      '-e',
      `fetch(${JSON.stringify(url)}).then(r => r.text()).then(t => process.stdout.write(t))`,
    ],
    { encoding: 'utf-8', timeout: 15000, stdio: ['ignore', 'pipe', 'ignore'] }
  );
}

/** Proxies are often written without a scheme, which fetch requires. */
function normalize(proxy: string): string {
  return /^https?:\/\//i.test(proxy) ? proxy : `http://${proxy}`;
}

/**
 * Arranges for fetch to go through the proxy this machine uses.
 *
 * Node only consults the proxy environment variables when told to, and only at
 * startup, so this re-executes the work in a child process when a proxy was
 * discovered rather than configured. Returns the environment a download should
 * run under.
 */
export function proxyEnvironment(proxy: string | null): NodeJS.ProcessEnv {
  if (!proxy) return process.env;

  return {
    ...process.env,
    NODE_USE_ENV_PROXY: '1',
    HTTPS_PROXY: proxy,
    HTTP_PROXY: proxy,
  };
}
