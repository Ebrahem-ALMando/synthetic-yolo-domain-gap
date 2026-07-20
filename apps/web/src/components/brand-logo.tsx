import Image from "next/image";
import { cn } from "@/src/lib/utils";

export function BrandLogo({ compact = false, className }: { compact?: boolean; className?: string }) {
  return (
    <div className={cn("relative overflow-hidden rounded-xl bg-white ring-1 ring-slate-200/80", compact ? "h-11 w-11" : "h-20 w-52", className)}>
      <Image
        src="/brand/synthdet-logo.png"
        alt="الشعار الرسمي لمنصة SynthDet"
        fill
        priority
        sizes={compact ? "44px" : "208px"}
        className="object-contain p-1.5"
      />
    </div>
  );
}
