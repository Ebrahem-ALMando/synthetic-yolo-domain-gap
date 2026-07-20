import type { Metadata } from "next";
import { RepositoryRoute } from "@/src/components/repository-route";
export const metadata: Metadata = { title: "مختبر الاستدلال" };
export default function Page() { return <RepositoryRoute path="/inference" />; }
