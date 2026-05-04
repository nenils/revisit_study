export const PREFIX = import.meta.env.PROD
  ? import.meta.env.VITE_BASE_PATH || '/'
  : '/';

export function sameOriginHref(path: string) {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;

  if (typeof window === 'undefined') {
    return normalizedPath;
  }

  return `${window.location.origin}${normalizedPath}`;
}
