import type { Metadata } from "next";
import { RepositoryRoute } from "@/src/components/repository-route";
export const metadata: Metadata = { title: "التجارب", description: "الأنظمة التجريبية الخمسة" };
export default function Page() { return <RepositoryRoute path="/experiments" />; }
