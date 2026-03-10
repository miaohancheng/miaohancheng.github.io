'use strict';

function pad(value) {
  return String(value).padStart(2, '0');
}

function dateParts(postDate) {
  if (postDate && typeof postDate.format === 'function') {
    return {
      year: postDate.format('YYYY'),
      month: postDate.format('MM'),
      day: postDate.format('DD')
    };
  }

  const parsed = new Date(postDate);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }

  return {
    year: String(parsed.getFullYear()),
    month: pad(parsed.getMonth() + 1),
    day: pad(parsed.getDate())
  };
}

function slugFromPath(postPath) {
  return String(postPath || '')
    .replace(/index\.html$/, '')
    .replace(/\/$/, '')
    .split('/')
    .filter(Boolean)
    .pop();
}

function stripIndex(postPath) {
  return String(postPath || '').replace(/index\.html$/, '');
}

function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function redirectDocument(targetPath, absoluteTarget) {
  const escapedTarget = escapeHtml(targetPath);
  const escapedAbsoluteTarget = escapeHtml(absoluteTarget);

  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Redirecting…</title>
    <meta name="robots" content="noindex, follow">
    <link rel="canonical" href="${escapedAbsoluteTarget}">
    <meta http-equiv="refresh" content="0; url=${escapedTarget}">
    <script>window.location.replace(${JSON.stringify(targetPath)});</script>
  </head>
  <body>
    <p>Redirecting to <a href="${escapedTarget}">${escapedTarget}</a>.</p>
  </body>
</html>`;
}

hexo.extend.generator.register('legacy_redirects', function(locals) {
  const root = (hexo.config.root || '/').replace(/\/?$/, '/');
  const siteUrl = String(hexo.config.url || '').replace(/\/$/, '');

  return locals.posts.toArray().map(function(post) {
    const parts = dateParts(post.date);
    const slug = slugFromPath(post.path);

    if (!parts || !slug) {
      return null;
    }

    const legacyPath = `${parts.year}/${parts.month}/${parts.day}/${slug}/index.html`;
    if (legacyPath === post.path) {
      return null;
    }

    const targetPath = `${root}${stripIndex(post.path)}`.replace(/\/{2,}/g, '/');
    const absoluteTarget = `${siteUrl}${targetPath}`;

    return {
      path: legacyPath,
      data: redirectDocument(targetPath, absoluteTarget),
      layout: false
    };
  }).filter(Boolean);
});
