import type { Metadata } from "next";
import { RepositoryRoute } from "@/src/components/repository-route";
export const metadata: Metadata = { title: "تفاصيل التجربة" };
export default async function Page({ params }: { params: Promise<{ regime: string }> }) { const { regime } = await params; return <RepositoryRoute path={`/experiments/${regime}`} />; }
