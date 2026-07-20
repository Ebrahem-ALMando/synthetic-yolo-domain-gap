import type { Metadata } from "next";
import { RepositoryRoute } from "@/src/components/repository-route";
export const metadata: Metadata = { title: "حالة النظام" };
export default function Page() { return <RepositoryRoute path="/system" />; }
