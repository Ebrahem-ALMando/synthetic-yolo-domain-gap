import { BrandLogo } from "@/src/components/brand-logo";

export default function Loading() {
  return <div role="status" aria-label="جارٍ التحميل" className="space-y-5"><div className="flex h-24 items-center gap-4 rounded-2xl border bg-card p-4"><BrandLogo className="h-16 w-40" /><div className="h-10 flex-1 animate-pulse rounded-xl bg-muted" /></div><div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">{Array.from({ length: 8 }).map((_, index) => <div key={index} className="h-36 animate-pulse rounded-2xl bg-muted" />)}</div></div>;
}
