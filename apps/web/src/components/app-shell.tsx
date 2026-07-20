"use client";

import * as Dialog from "@radix-ui/react-dialog";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import * as Tooltip from "@radix-ui/react-tooltip";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bell,
  ChevronLeft,
  ChevronsLeft,
  ChevronsRight,
  Command,
  Menu,
  Moon,
  Search,
  Sun,
  UserRound,
  X,
} from "lucide-react";
import { useTheme } from "next-themes";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { BrandLogo } from "@/src/components/brand-logo";
import { Button, StatusBadge } from "@/src/components/ui";
import { appRoutes, routeForPath } from "@/src/lib/routes";
import { cn } from "@/src/lib/utils";

function Navigation({ collapsed = false, onNavigate }: { collapsed?: boolean; onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav aria-label="التنقل الرئيسي" className="space-y-1">
      {appRoutes.map((route) => {
        const active = route.href === "/" ? pathname === "/" : pathname.startsWith(route.href);
        const Icon = route.icon;
        const link = <Link href={route.href} onClick={onNavigate} aria-current={active ? "page" : undefined} className={cn("group flex h-11 items-center gap-3 rounded-xl px-3 text-sm font-semibold transition", active ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:bg-muted hover:text-foreground", collapsed && "justify-center px-0")}><Icon className="h-[18px] w-[18px] shrink-0" /><span className={cn(collapsed && "sr-only")}>{route.title}</span>{!collapsed && active && <ChevronLeft className="mr-auto h-4 w-4" />}</Link>;
        return collapsed ? <Tooltip.Root key={route.href}><Tooltip.Trigger asChild>{link}</Tooltip.Trigger><Tooltip.Portal><Tooltip.Content side="left" className="z-50 rounded-lg bg-foreground px-3 py-1.5 text-xs text-background shadow-lg">{route.title}<Tooltip.Arrow className="fill-foreground" /></Tooltip.Content></Tooltip.Portal></Tooltip.Root> : <div key={route.href}>{link}</div>;
      })}
    </nav>
  );
}

function ThemeButton() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <span className="h-10 w-10" />;
  const dark = resolvedTheme === "dark";
  return <Button variant="ghost" className="w-10 px-0" aria-label={dark ? "تفعيل الوضع الفاتح" : "تفعيل الوضع الداكن"} onClick={() => setTheme(dark ? "light" : "dark")}>{dark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}</Button>;
}

function CommandPalette({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const [query, setQuery] = useState("");
  const router = useRouter();
  const results = useMemo(() => appRoutes.filter((route) => `${route.title} ${route.description}`.includes(query.trim())), [query]);
  return <Dialog.Root open={open} onOpenChange={onOpenChange}><Dialog.Portal><Dialog.Overlay className="fixed inset-0 z-50 bg-slate-950/45 backdrop-blur-[2px]" /><Dialog.Content className="fixed left-1/2 top-[15%] z-50 w-[min(92vw,38rem)] -translate-x-1/2 overflow-hidden rounded-2xl border bg-card shadow-lift"><Dialog.Title className="sr-only">البحث السريع</Dialog.Title><div className="flex items-center gap-3 border-b px-4"><Search className="h-5 w-5 text-muted-foreground" /><input autoFocus value={query} onChange={(event) => setQuery(event.target.value)} placeholder="ابحث عن صفحة أو مفهوم…" className="h-14 w-full bg-transparent text-sm outline-none" /><kbd className="rounded border bg-muted px-1.5 py-0.5 text-[10px]">Esc</kbd></div><div className="max-h-80 overflow-y-auto p-2">{results.map((route) => <button key={route.href} onClick={() => { router.push(route.href); onOpenChange(false); }} className="flex w-full items-center gap-3 rounded-xl p-3 text-right hover:bg-muted"><route.icon className="h-5 w-5 text-primary" /><span><b className="block text-sm">{route.title}</b><span className="text-xs text-muted-foreground">{route.description}</span></span></button>)}{results.length === 0 && <p className="p-8 text-center text-sm text-muted-foreground">لا توجد نتائج مطابقة.</p>}</div></Dialog.Content></Dialog.Portal></Dialog.Root>;
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const route = routeForPath(pathname);
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [commandOpen, setCommandOpen] = useState(false);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") { event.preventDefault(); setCommandOpen(true); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
  return (
    <div className="min-h-screen">
      <aside className={cn("fixed inset-y-0 right-0 z-40 hidden border-l bg-card/95 px-3 py-4 shadow-sm backdrop-blur lg:flex lg:flex-col", collapsed ? "w-[76px]" : "w-[248px]")}>
        <div className={cn("mb-5 flex items-center", collapsed ? "justify-center" : "justify-between")}><BrandLogo compact={collapsed} />{!collapsed && <button onClick={() => setCollapsed(true)} aria-label="طي القائمة" className="rounded-lg p-2 text-muted-foreground hover:bg-muted"><ChevronsRight className="h-4 w-4" /></button>}</div>
        <div className="scrollbar-thin flex-1 overflow-y-auto"><Navigation collapsed={collapsed} /></div>
        <div className="mt-3 border-t pt-3">{collapsed ? <button onClick={() => setCollapsed(false)} aria-label="توسيع القائمة" className="grid h-10 w-full place-items-center rounded-xl hover:bg-muted"><ChevronsLeft className="h-4 w-4" /></button> : <div className="rounded-xl bg-gradient-to-br from-primary/10 to-accent/10 p-3"><p className="text-xs font-bold">الحالة العلمية</p><div className="mt-2"><StatusBadge status="success" label="الحملة النهائية مختومة" /></div></div>}</div>
      </aside>

      <div className={cn("min-w-0 max-w-full transition-[padding] duration-200 lg:pr-[248px]", collapsed && "lg:pr-[76px]")}>
        <header className="sticky top-0 z-30 flex h-16 min-w-0 items-center gap-2 border-b bg-background/88 px-4 backdrop-blur-md sm:px-6">
          <Dialog.Root open={mobileOpen} onOpenChange={setMobileOpen}><Dialog.Trigger asChild><Button variant="ghost" className="w-10 px-0 lg:hidden" aria-label="فتح التنقل"><Menu className="h-5 w-5" /></Button></Dialog.Trigger><Dialog.Portal><Dialog.Overlay className="fixed inset-0 z-50 bg-slate-950/45" /><Dialog.Content className="fixed inset-y-0 right-0 z-50 w-[min(88vw,20rem)] border-l bg-card p-4 shadow-lift"><Dialog.Title className="sr-only">التنقل الرئيسي</Dialog.Title><div className="mb-5 flex items-center justify-between"><BrandLogo /><Dialog.Close asChild><Button variant="ghost" className="w-10 px-0" aria-label="إغلاق التنقل"><X /></Button></Dialog.Close></div><Navigation onNavigate={() => setMobileOpen(false)} /></Dialog.Content></Dialog.Portal></Dialog.Root>
          <div className="hidden items-center gap-2 text-sm sm:flex"><Link href="/" className="text-muted-foreground hover:text-foreground">SynthDet</Link><ChevronLeft className="h-4 w-4 text-muted-foreground" /><span className="font-bold">{route?.title ?? "صفحة المشروع"}</span></div>
          <button onClick={() => setCommandOpen(true)} className="mr-auto flex h-10 min-w-10 items-center gap-2 rounded-xl border bg-card px-3 text-sm text-muted-foreground shadow-sm hover:bg-muted" aria-label="فتح البحث"><Search className="h-4 w-4" /><span className="hidden md:inline">بحث سريع</span><kbd className="hidden rounded border bg-muted px-1.5 py-0.5 text-[10px] lg:inline">Ctrl K</kbd></button>
          <ThemeButton />
          <DropdownMenu.Root><DropdownMenu.Trigger asChild><Button variant="ghost" className="relative w-10 px-0" aria-label="الإشعارات"><Bell className="h-5 w-5" /><span className="absolute left-2 top-2 h-2 w-2 rounded-full bg-success ring-2 ring-background" /></Button></DropdownMenu.Trigger><DropdownMenu.Portal><DropdownMenu.Content align="end" className="z-50 w-72 rounded-xl border bg-card p-2 shadow-lift"><DropdownMenu.Label className="p-2 text-sm font-bold">الإشعارات</DropdownMenu.Label><div className="rounded-lg bg-emerald-500/8 p-3 text-xs leading-6">اكتملت الأنظمة الخمسة وحملة الاختبار المحمية. النموذج الموصى به هو real_only.</div></DropdownMenu.Content></DropdownMenu.Portal></DropdownMenu.Root>
          <DropdownMenu.Root><DropdownMenu.Trigger asChild><Button variant="ghost" className="w-10 px-0" aria-label="قائمة المشروع"><UserRound className="h-5 w-5" /></Button></DropdownMenu.Trigger><DropdownMenu.Portal><DropdownMenu.Content align="end" className="z-50 w-56 rounded-xl border bg-card p-2 shadow-lift"><DropdownMenu.Label className="p-2"><b className="block text-sm">مشروع SynthDet</b><span className="text-xs text-muted-foreground">مشروع جامعي — رؤية حاسوبية</span></DropdownMenu.Label><DropdownMenu.Separator className="my-1 h-px bg-border" /><DropdownMenu.Item asChild><Link href="/about" className="block cursor-pointer rounded-lg p-2 text-sm outline-none hover:bg-muted">حول المشروع</Link></DropdownMenu.Item><DropdownMenu.Item asChild><Link href="/system" className="block cursor-pointer rounded-lg p-2 text-sm outline-none hover:bg-muted">حالة النظام</Link></DropdownMenu.Item></DropdownMenu.Content></DropdownMenu.Portal></DropdownMenu.Root>
        </header>
        <AnimatePresence mode="wait"><motion.main key={pathname} initial={false} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: .2 }} className="mx-auto min-w-0 w-full max-w-[1600px] p-4 sm:p-6 lg:p-8">{children}</motion.main></AnimatePresence>
      </div>
      <CommandPalette open={commandOpen} onOpenChange={setCommandOpen} />
      <button onClick={() => setCommandOpen(true)} className="fixed bottom-5 left-5 z-20 hidden h-11 items-center gap-2 rounded-full bg-foreground px-4 text-xs font-bold text-background shadow-lift xl:flex"><Command className="h-4 w-4" />لوحة الأوامر</button>
    </div>
  );
}
