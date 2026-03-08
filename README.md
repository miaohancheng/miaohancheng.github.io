# miaohancheng.com

Personal Hexo site for Miao Hancheng.

## Local development

Install dependencies:

```bash
npm install
```

Start the local server:

```bash
npm run server
```

Build the static site:

```bash
npm run build
```

## Deployment

Pushes to `main` trigger the GitHub Actions workflow in
`.github/workflows/pages.yml`, which builds the site, writes the
`miaohancheng.com` CNAME, and deploys the generated `public/` directory to
GitHub Pages.
