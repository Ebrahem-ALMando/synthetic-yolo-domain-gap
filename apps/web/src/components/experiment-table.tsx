"use client";

import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { ArrowUpDown, ExternalLink, Search, SlidersHorizontal } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import { Button, StatusBadge, TechnicalValue } from "@/src/components/ui";
import type { ExperimentRegime } from "@/src/types/domain";
import { formatNumber } from "@/src/lib/utils";

const helper = createColumnHelper<ExperimentRegime>();
const columns = [
  helper.accessor("nameAr", {
    header: "التجربة",
    cell: (info) => (
      <Link className="font-bold text-primary hover:underline" href={`/experiments/${info.row.original.id}`}>
        {info.getValue()}
      </Link>
    ),
  }),
  helper.accessor("realCount", { header: "حقيقي", cell: (info) => formatNumber.format(info.getValue()) }),
  helper.accessor("syntheticCount", { header: "اصطناعي", cell: (info) => formatNumber.format(info.getValue()) }),
  helper.accessor("validationCount", { header: "التحقق المشترك", cell: (info) => formatNumber.format(info.getValue()) }),
  helper.accessor("status", { header: "الحالة", cell: (info) => <StatusBadge status={info.getValue()} /> }),
  helper.accessor("manifestHash", { header: "هوية البيان", cell: (info) => <TechnicalValue value={info.getValue()} /> }),
  helper.display({
    id: "actions",
    header: "",
    cell: (info) => (
      <Button asChild variant="ghost" className="w-9 px-0">
        <Link href={`/experiments/${info.row.original.id}`} aria-label={`فتح ${info.row.original.nameAr}`}>
          <ExternalLink className="h-4 w-4" />
        </Link>
      </Button>
    ),
  }),
];

export function ExperimentTable({ data }: { data: ExperimentRegime[] }) {
  const [filter, setFilter] = useState("");
  const table = useReactTable({
    data,
    columns,
    state: { globalFilter: filter },
    onGlobalFilterChange: setFilter,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });
  return (
    <div className="min-w-0 max-w-full overflow-hidden">
      <div className="mb-4 flex flex-col gap-2 sm:flex-row">
        <label className="relative min-w-0 flex-1">
          <span className="sr-only">بحث في التجارب</span>
          <Search className="absolute right-3 top-3 h-4 w-4 text-muted-foreground" />
          <input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="ابحث في التجارب…" className="h-10 w-full rounded-xl border bg-background pr-9 text-sm outline-none" />
        </label>
        <Button variant="outline" disabled><SlidersHorizontal className="h-4 w-4" />إظهار الأعمدة</Button>
        <Button variant="outline" disabled>تصدير — بعد النتائج</Button>
      </div>
      <div className="max-w-full overflow-x-auto overscroll-x-contain rounded-xl border">
        <table className="w-full min-w-[820px] text-sm">
          <thead className="bg-muted/60">
            <tr>{table.getHeaderGroups()[0].headers.map((header) => <th key={header.id} className="px-4 py-3 text-right text-xs font-extrabold text-muted-foreground"><button className="inline-flex items-center gap-1" onClick={header.column.getToggleSortingHandler()}>{flexRender(header.column.columnDef.header, header.getContext())}{header.column.getCanSort() && <ArrowUpDown className="h-3 w-3" />}</button></th>)}</tr>
          </thead>
          <tbody>{table.getRowModel().rows.map((row) => <tr key={row.id} className="border-t transition hover:bg-muted/35">{row.getVisibleCells().map((cell) => <td key={cell.id} className="px-4 py-3">{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>)}</tr>)}</tbody>
        </table>
        {table.getRowModel().rows.length === 0 && <p className="p-8 text-center text-muted-foreground">لا توجد تجارب مطابقة.</p>}
      </div>
    </div>
  );
}
