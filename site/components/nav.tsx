"use client";

import { useEffect, useState } from "react";
import { outline } from "@/content/site";

/* A fixed left-hand outline, like a document table of contents: section titles at the first
   level, subsections indented beneath them. The entry for what is on screen is darkened. */
export function Rail() {
  const [active, setActive] = useState<string>("");

  useEffect(() => {
    const ids = outline.flatMap((o) => [o.id, ...(o.children ?? []).map((c) => c.id)]);
    const els = ids.map((id) => document.getElementById(id)).filter((el): el is HTMLElement => el !== null);
    const update = () => {
      const line = window.innerHeight * 0.3;
      let current = "";
      for (const el of els) {
        if (el.getBoundingClientRect().top <= line) current = el.id;
      }
      setActive(current);
    };
    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, []);

  const cls = (id: string, sub = false) =>
    `block leading-snug transition-colors hover:text-ink ${sub ? "text-[0.8rem]" : "text-[0.85rem]"} ${active === id ? "text-ink" : "text-muted"}`;

  return (
    <nav aria-label="Contents" className="fixed left-8 top-1/2 z-10 hidden w-44 -translate-y-1/2 xl:block">
      <p className="text-[0.85rem] text-ink">Contents</p>
      <ol className="mt-3 space-y-2">
        {outline.map((o) => (
          <li key={o.id}>
            <a href={`#${o.id}`} className={cls(o.id)}>
              {o.label}
            </a>
            {o.children ? (
              <ol className="mt-1.5 space-y-1.5 pl-4">
                {o.children.map((c) => (
                  <li key={c.id}>
                    <a href={`#${c.id}`} className={cls(c.id, true)}>
                      {c.label}
                    </a>
                  </li>
                ))}
              </ol>
            ) : null}
          </li>
        ))}
      </ol>
    </nav>
  );
}
