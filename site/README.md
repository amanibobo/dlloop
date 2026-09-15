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
environment variables needed.
