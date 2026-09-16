# LensCraft case-study site

Next.js (App Router) + Tailwind. Single long page; content lives in `content/site.ts`, figures in
`public/figures` (copied from `../reports/final`), demo recordings go in `public/videos` (see the
README there for the expected file names; missing files show a placeholder).

```sh
npm install
npm run dev      # http://localhost:3000
npm run build
```

Deploy on Vercel: import the repo, set **Root Directory** to `site`, framework preset Next.js. No
environment variables needed (`NEXT_PUBLIC_SITE_URL` fixes social-preview links).

Interactive pieces: `public/marimo/{uncertainty,passk,playground}` are marimo WASM exports of the notebooks in
`../notebooks` (see the root README to regenerate); all three run in the visitor's browser, nothing
runs on a server. `public/media/` takes hand-drawn diagrams and screenshots, `public/videos/`
the demo recordings; both show placeholders until the files exist.
