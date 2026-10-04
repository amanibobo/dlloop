/* The looping gravitational-lensing animation (public/lens/index.html, anime.js), embedded in
   place of the hero image. Self-contained: its script and texture ship with the site. */
export function LensAnimation() {
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-white">
      <iframe src="/lens/index.html" title="Gravitational lensing: light from a distant galaxy bends around a cluster toward Earth" loading="eager" className="block aspect-[2296/1607] w-full" scrolling="no" />
    </div>
  );
}
