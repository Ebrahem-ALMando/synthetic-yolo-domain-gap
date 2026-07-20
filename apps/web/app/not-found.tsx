import Link from "next/link";
import { Button, EmptyState } from "@/src/components/ui";

export default function NotFound() {
  return <div className="space-y-4"><EmptyState title="الصفحة غير موجودة" description="المسار المطلوب ليس جزءًا من بنية Sprint 6A." /><Button asChild><Link href="/">العودة إلى النظرة العامة</Link></Button></div>;
}
