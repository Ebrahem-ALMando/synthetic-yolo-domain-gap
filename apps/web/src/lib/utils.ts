import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const formatNumber = new Intl.NumberFormat("ar-SY");
export const formatPercent = new Intl.NumberFormat("ar-SY", {
  style: "percent",
  maximumFractionDigits: 1,
});
