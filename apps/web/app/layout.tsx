import type { Metadata } from "next";
import { Tajawal } from "next/font/google";
import "./globals.css";
import { AppShell } from "@/src/components/app-shell";
import { Providers } from "@/src/components/providers";

const tajawal = Tajawal({
  subsets: ["arabic"],
  // Tajawal does not publish a 600 face; CSS 600 text resolves between its 500/700 faces.
  weight: ["300", "400", "500", "700", "800"],
  display: "swap",
  variable: "--font-tajawal",
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000"),
  title: { default: "SynthDet | سينث دِت", template: "%s | SynthDet" },
  description: "منصة تحليل البيانات الاصطناعية وكشف الأجسام",
  icons: { icon: "/brand/synthdet-logo.png", apple: "/brand/synthdet-logo.png" },
  openGraph: { title: "SynthDet | سينث دِت", description: "رؤية ذكية. بيانات اصطناعية. كشف أدق.", images: ["/brand/synthdet-logo.png"], locale: "ar_SY", type: "website" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ar" dir="rtl" suppressHydrationWarning><body className={tajawal.className}><Providers><AppShell>{children}</AppShell></Providers></body></html>;
}
