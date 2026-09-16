import type { Metadata } from "next";
import { Geist_Mono, Instrument_Sans } from "next/font/google";
import "./globals.css";

const sans = Instrument_Sans({ variable: "--font-sans-face", subsets: ["latin"], weight: ["400", "500", "600"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  title: "Dark Matter by Feedback: Uncertainty-Driven Simulation for Strong-Lensing Detectors",
  description:
    "An agent that simulates gravitational lenses, trains dark-matter substructure detectors on them, and uses what the models can't classify to decide what to simulate next.",
  openGraph: {
    title: "Dark Matter by Feedback: Uncertainty-Driven Simulation for Strong-Lensing Detectors",
    description: "Closing the simulate-train-resimulate loop. A case study built on Pydantic AI, Lenstronomy and Modal.",
    images: ["/figures/hero_classes.png"],
  },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${sans.variable} ${geistMono.variable} h-full`}>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
