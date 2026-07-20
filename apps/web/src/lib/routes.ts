import type { LucideIcon } from "lucide-react";
import {
  Activity,
  BarChart3,
  BookOpen,
  BrainCircuit,
  Database,
  FileBarChart,
  FlaskConical,
  Gauge,
  Images,
  Microscope,
  ScanSearch,
  ShieldCheck,
} from "lucide-react";

export interface AppRoute {
  href: string;
  title: string;
  shortTitle: string;
  description: string;
  icon: LucideIcon;
}

export const appRoutes: AppRoute[] = [
  { href: "/", title: "نظرة عامة", shortTitle: "الرئيسية", description: "ملخص المشروع والعقود العلمية المجمدة", icon: Gauge },
  { href: "/experiments", title: "التجارب", shortTitle: "التجارب", description: "مقارنة أنظمة المزج الخمسة وتكويناتها", icon: FlaskConical },
  { href: "/datasets/real", title: "البيانات الحقيقية", shortTitle: "الحقيقية", description: "Split V2 وسلامة المصدر والاختبار المحمي", icon: Database },
  { href: "/datasets/synthetic", title: "البيانات الاصطناعية", shortTitle: "الاصطناعية", description: "بنك الأجسام وخط توليد النسخ واللصق", icon: Images },
  { href: "/training", title: "التدريب", shortTitle: "التدريب", description: "حالة التنفيذ الخارجي للأنظمة الخمسة", icon: Activity },
  { href: "/evaluation", title: "التقييم والمقارنة", shortTitle: "التقييم", description: "مساحة مقفلة للنتائج العلمية النهائية", icon: BarChart3 },
  { href: "/analysis", title: "تحليل الأخطاء", shortTitle: "الأخطاء", description: "معرض تحليلي لحالات النجاح والإخفاق", icon: Microscope },
  { href: "/inference", title: "مختبر الاستدلال", shortTitle: "الاستدلال", description: "واجهة جاهزة لربط النماذج بعد اعتمادها", icon: ScanSearch },
  { href: "/reports", title: "التقارير", shortTitle: "التقارير", description: "مخرجات قابلة للتتبع وحالات التصدير", icon: FileBarChart },
  { href: "/reproducibility", title: "قابلية إعادة الإنتاج", shortTitle: "إعادة الإنتاج", description: "الهويات والبذور والبيئة وسجل التدقيق", icon: ShieldCheck },
  { href: "/system", title: "حالة النظام", shortTitle: "النظام", description: "صحة الواجهة والتكاملات المستقبلية", icon: BrainCircuit },
  { href: "/about", title: "حول المشروع", shortTitle: "حول", description: "الهدف الأكاديمي والمنهجية والحدود", icon: BookOpen },
];

export function normalizePath(slug?: string[]) {
  return slug?.length ? `/${slug.join("/")}` : "/";
}

export function routeForPath(path: string) {
  if (path.startsWith("/experiments/")) return appRoutes[1];
  return appRoutes.find((route) => route.href === path);
}
