"use client";

import { TriangleAlert } from "lucide-react";
import { Button, Card } from "@/src/components/ui";

export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return <Card className="grid min-h-[60vh] place-items-center p-8 text-center"><div><TriangleAlert className="mx-auto h-12 w-12 text-destructive" /><h1 className="mt-4 text-xl font-extrabold">تعذر تحميل هذه المساحة</h1><p className="mt-2 text-sm text-muted-foreground">لم يتم استخدام بيانات بديلة تلقائيًا. راجع وضع البيانات ثم أعد المحاولة.</p><Button onClick={reset} className="mt-5">إعادة المحاولة</Button></div></Card>;
}
