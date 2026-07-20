"use client";

import { Slot } from "@radix-ui/react-slot";
import { Check, Copy, Info, LockKeyhole, TriangleAlert, XCircle } from "lucide-react";
import { useState } from "react";
import type { ScientificStatus } from "@/src/types/domain";
import { cn, formatNumber } from "@/src/lib/utils";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("rounded-2xl border bg-card text-card-foreground shadow-soft", className)} {...props} />;
}

export function Button({ asChild, variant = "primary", className, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { asChild?: boolean; variant?: "primary" | "secondary" | "ghost" | "outline" | "danger" }) {
  const Component = asChild ? Slot : "button";
  return (
    <Component
      className={cn(
        "inline-flex h-10 items-center justify-center gap-2 rounded-xl px-4 text-sm font-bold transition duration-200 disabled:pointer-events-none disabled:opacity-45",
        variant === "primary" && "bg-primary text-primary-foreground shadow-sm hover:-translate-y-0.5 hover:bg-primary/90",
        variant === "secondary" && "bg-secondary text-secondary-foreground hover:bg-secondary/75",
        variant === "ghost" && "hover:bg-muted",
        variant === "outline" && "border bg-card hover:bg-muted",
        variant === "danger" && "bg-destructive text-destructive-foreground",
        className,
      )}
      {...props}
    />
  );
}

const statusStyles: Record<ScientificStatus, string> = {
  completed: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  in_progress: "bg-blue-500/10 text-blue-700 dark:text-blue-300",
  awaiting_results: "bg-amber-500/10 text-amber-700 dark:text-amber-300",
  protected: "bg-violet-500/10 text-violet-700 dark:text-violet-300",
  frozen: "bg-cyan-500/10 text-cyan-700 dark:text-cyan-300",
  success: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  warning: "bg-amber-500/10 text-amber-700 dark:text-amber-300",
  failed: "bg-red-500/10 text-red-700 dark:text-red-300",
  unavailable: "bg-slate-500/10 text-slate-600 dark:text-slate-300",
};

const statusLabels: Record<ScientificStatus, string> = {
  completed: "مكتمل",
  in_progress: "قيد التنفيذ",
  awaiting_results: "بانتظار النتائج",
  protected: "محمي",
  frozen: "مجمّد",
  success: "ناجح",
  warning: "تحذير",
  failed: "فشل",
  unavailable: "غير متاح",
};

export function StatusBadge({ status, label }: { status: ScientificStatus; label?: string }) {
  const Icon = status === "failed" ? XCircle : status === "warning" || status === "awaiting_results" ? TriangleAlert : status === "protected" ? LockKeyhole : Check;
  return <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-bold", statusStyles[status])}><Icon className="h-3.5 w-3.5" aria-hidden />{label ?? statusLabels[status]}</span>;
}

export function PageHeader({ title, description, eyebrow, actions }: { title: string; description: string; eyebrow?: string; actions?: React.ReactNode }) {
  return (
    <header className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
      <div className="max-w-3xl">
        {eyebrow && <p className="mb-1 text-xs font-extrabold tracking-wide text-primary">{eyebrow}</p>}
        <h1 className="text-2xl font-extrabold tracking-tight sm:text-3xl">{title}</h1>
        <p className="mt-2 text-sm leading-7 text-muted-foreground sm:text-base">{description}</p>
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function MetricCard({ label, value, helper, icon: Icon, status }: { label: string; value: string | number; helper: string; icon: React.ComponentType<{ className?: string }>; status?: ScientificStatus }) {
  return (
    <Card className="group relative overflow-hidden p-4 transition duration-200 hover:-translate-y-0.5 hover:shadow-lift">
      <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-l from-primary via-accent to-sky-400 opacity-70" />
      <div className="flex items-start justify-between gap-3">
        <div><p className="text-xs font-semibold text-muted-foreground">{label}</p><p className="mt-2 text-2xl font-extrabold tabular-nums">{typeof value === "number" ? formatNumber.format(value) : value}</p></div>
        <span className="grid h-10 w-10 place-items-center rounded-xl bg-primary/8 text-primary"><Icon className="h-5 w-5" /></span>
      </div>
      <div className="mt-3 flex min-h-6 items-center justify-between gap-2 text-xs text-muted-foreground"><span>{helper}</span>{status && <StatusBadge status={status} />}</div>
    </Card>
  );
}

export function TechnicalValue({ value, copy = true, className }: { value: string; copy?: boolean; className?: string }) {
  const [copied, setCopied] = useState(false);
  async function handleCopy() {
    await navigator.clipboard.writeText(value);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  }
  return (
    <span className={cn("inline-flex max-w-full items-center gap-2 rounded-lg bg-muted px-2.5 py-1.5 text-xs", className)}>
      <bdi dir="ltr" className="technical-ltr truncate" title={value}>{value}</bdi>
      {copy && <button onClick={handleCopy} aria-label={copied ? "تم النسخ" : "نسخ القيمة"} className="shrink-0 text-muted-foreground hover:text-primary">{copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}</button>}
    </span>
  );
}

export function DemoDataBanner() {
  return <div role="status" data-testid="demo-data-banner" className="flex items-center gap-2 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm font-bold text-amber-900 dark:border-amber-700 dark:bg-amber-950/50 dark:text-amber-200"><TriangleAlert className="h-5 w-5 shrink-0" />بيانات تجريبية للعرض — ليست نتائج علمية نهائية</div>;
}

export function PendingScientificResults({ compact = false }: { compact?: boolean }) {
  return <div className={cn("grid place-items-center rounded-2xl border border-dashed bg-muted/35 text-center", compact ? "min-h-40 p-5" : "min-h-64 p-8")}><div><span className="mx-auto grid h-12 w-12 place-items-center rounded-full bg-primary/10 text-primary"><LockKeyhole /></span><h3 className="mt-3 font-extrabold">النتائج العلمية مقفلة</h3><p className="mt-1 max-w-md text-sm leading-6 text-muted-foreground">لن تظهر المقاييس النهائية قبل عودة نتائج التدريب وتنفيذ بروتوكول التقييم المحمي.</p></div></div>;
}

export function EmptyState({ title, description }: { title: string; description: string }) {
  return <div className="grid min-h-48 place-items-center rounded-2xl border border-dashed bg-muted/20 p-8 text-center"><div><Info className="mx-auto h-9 w-9 text-primary" /><h3 className="mt-3 font-bold">{title}</h3><p className="mt-1 text-sm text-muted-foreground">{description}</p></div></div>;
}

export function ProtectedTestBadge() {
  return <StatusBadge status="protected" label="اختبار محمي — 68 صورة" />;
}

export function RegimeCompositionBar({ real, synthetic, label = true }: { real: number; synthetic: number; label?: boolean }) {
  const total = real + synthetic || 1;
  const realWidth = (real / total) * 100;
  return <div><div className="flex h-2.5 overflow-hidden rounded-full bg-muted" aria-label={`حقيقي ${real}، اصطناعي ${synthetic}`}><span className="bg-primary" style={{ width: `${realWidth}%` }} /><span className="bg-accent" style={{ width: `${100 - realWidth}%` }} /></div>{label && <div className="mt-2 flex justify-between text-[11px] text-muted-foreground"><span>حقيقي {formatNumber.format(real)}</span><span>اصطناعي {formatNumber.format(synthetic)}</span></div>}</div>;
}

export function IdentityCard({ title, value, status = "frozen" }: { title: string; value: string; status?: ScientificStatus }) {
  return <Card className="p-4"><div className="mb-3 flex items-center justify-between"><h3 className="text-sm font-bold">{title}</h3><StatusBadge status={status} /></div><TechnicalValue value={value} className="w-full justify-between" /></Card>;
}
