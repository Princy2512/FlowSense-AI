import { useEffect, useMemo, useState } from "react";

type Metadata = {
  categorical: {
    RoadType: string[];
    Weather: string[];
    LargeVehicles: string[];
    Landmarks: string[];
  };
  geohashTop: string[];
  dayRange: { min: number; max: number };
  lanesOptions: number[];
  timestampOptions: string[];
  temperatureRange: { min: number; max: number };
  cvR2?: number;
};

type FormState = {
  geohash: string;
  day: number;
  timestamp: string;
  RoadType: string;
  NumberofLanes: number;
  LargeVehicles: string;
  Landmarks: string;
  Temperature: number;
  Weather: string;
};

function classNames(...xs: Array<string | false | undefined>) {
  return xs.filter(Boolean).join(" ");
}

const BRAND = "FlowSense AI";

function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1 text-xs text-slate-200">
      <span className="text-slate-400">{label}</span> <span className="font-semibold">{value}</span>
    </div>
  );
}

type SavedEstimate = {
  id: string;
  createdAt: number;
  form: FormState;
  demand: number;
};

function useLocalStorageState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // ignore
    }
  }, [key, value]);

  return [value, setValue] as const;
}

export default function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [loadingMeta, setLoadingMeta] = useState(true);
  const [metaError, setMetaError] = useState<string | null>(null);

  const [form, setForm] = useState<FormState | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [saved, setSaved] = useLocalStorageState<SavedEstimate[]>(`${BRAND}:saved`, []);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoadingMeta(true);
        const res = await fetch("/api/metadata");
        const json = await res.json();
        if (!res.ok) throw new Error(json?.message || "Failed to load metadata");
        if (!alive) return;
        setMetadata(json);
      } catch (e: any) {
        if (!alive) return;
        setMetaError(String(e?.message ?? e));
      } finally {
        if (alive) setLoadingMeta(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!metadata) return;
    const day = Math.max(metadata.dayRange.min, Math.min(metadata.dayRange.max, metadata.dayRange.min));
    const ts = metadata.timestampOptions?.[0] ?? "0:0";
    const road = metadata.categorical.RoadType?.[0] ?? "Residential";
    const weather = metadata.categorical.Weather?.[0] ?? "Sunny";
    const lv = metadata.categorical.LargeVehicles?.[0] ?? "Allowed";
    const lm = metadata.categorical.Landmarks?.[0] ?? "No";
    const lane = metadata.lanesOptions?.[0] ?? 1;
    const temp = Number.isFinite(metadata.temperatureRange?.min) ? metadata.temperatureRange.min : 20;
    const gh = metadata.geohashTop?.[0] ?? "";
    setForm({
      geohash: gh,
      day,
      timestamp: ts,
      RoadType: road,
      NumberofLanes: lane,
      LargeVehicles: lv,
      Landmarks: lm,
      Temperature: temp,
      Weather: weather,
    });
  }, [metadata]);

  const canPredict = useMemo(() => {
    if (!form) return false;
    return (
      form.geohash.trim().length >= 5 &&
      Number.isFinite(form.day) &&
      String(form.timestamp).includes(":") &&
      String(form.RoadType).length > 0 &&
      Number.isFinite(form.NumberofLanes) &&
      String(form.LargeVehicles).length > 0 &&
      String(form.Landmarks).length > 0 &&
      Number.isFinite(form.Temperature) &&
      String(form.Weather).length > 0
    );
  }, [form]);

  useEffect(() => {
    if (!toast) return;
    const t = window.setTimeout(() => setToast(null), 2500);
    return () => window.clearTimeout(t);
  }, [toast]);

  async function onPredict() {
    if (!form) return;
    setPredicting(true);
    setPredictError(null);
    setPrediction(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.error || json?.message || "Prediction failed");
      const demand = Number(json?.demand);
      setPrediction(demand);
      if (Number.isFinite(demand)) {
        const item: SavedEstimate = {
          id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
          createdAt: Date.now(),
          form,
          demand,
        };
        setSaved((prev) => [item, ...prev].slice(0, 12));
      }
    } catch (e: any) {
      setPredictError(String(e?.message ?? e));
    } finally {
      setPredicting(false);
    }
  }

  function loadSaved(item: SavedEstimate) {
    setForm(item.form);
    setPrediction(item.demand);
    setPredictError(null);
    setToast("Loaded saved estimate");
  }

  function clearSaved() {
    setSaved([]);
    setToast("History cleared");
  }

  return (
    <div className="min-h-screen bg-grid">
      <div className="sticky top-0 z-20 border-b border-slate-800/80 glass">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-sky-400/90 to-emerald-300/90 shadow-soft" />
            <div>
              <div className="text-sm font-semibold tracking-tight">{BRAND}</div>
              <div className="text-[11px] text-slate-400">Demand Console</div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <StatPill label="Status" value={loadingMeta ? "Syncing…" : metaError ? "Needs setup" : "Online"} />
            <button
              onClick={() => {
                navigator.clipboard
                  .writeText(JSON.stringify(form ?? {}, null, 2))
                  .then(() => setToast("Copied inputs"))
                  .catch(() => setToast("Copy failed"));
              }}
              className="hidden sm:inline-flex items-center rounded-xl border border-slate-800 bg-slate-950/50 px-3 py-2 text-xs font-semibold text-slate-200 hover:border-slate-700"
              type="button"
            >
              Copy inputs
            </button>
          </div>
        </div>
      </div>

      {toast ? (
        <div className="fixed bottom-5 left-1/2 z-50 -translate-x-1/2">
          <div className="rounded-full border border-slate-800 bg-slate-950/80 px-4 py-2 text-xs text-slate-200 shadow-soft">
            {toast}
          </div>
        </div>
      ) : null}

      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-4 py-8 lg:grid-cols-12">
        <aside className="lg:col-span-4">
          <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-5 shadow-soft">
            <div className="flex items-start justify-between">
              <div>
                <div className="text-xs text-slate-400">Workspace</div>
                <div className="mt-1 text-lg font-semibold">Saved estimates</div>
              </div>
              <button
                onClick={clearSaved}
                className="rounded-xl border border-slate-800 bg-slate-950/40 px-3 py-2 text-xs font-semibold text-slate-200 hover:border-slate-700"
                type="button"
              >
                Clear
              </button>
            </div>

            <div className="mt-4 space-y-2">
              {saved.length === 0 ? (
                <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4 text-sm text-slate-300">
                  No history yet.
                </div>
              ) : (
                saved.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => loadSaved(item)}
                    className="w-full rounded-xl border border-slate-800 bg-slate-950/40 p-3 text-left text-sm text-slate-200 hover:border-slate-700"
                    type="button"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="font-semibold tabular-nums">{item.demand.toFixed(6)}</div>
                      <div className="text-[11px] text-slate-400">
                        {new Date(item.createdAt).toLocaleString()}
                      </div>
                    </div>
                    <div className="mt-2 text-[11px] text-slate-400">
                      {String(item.form.geohash).slice(0, 7)} • {item.form.timestamp} • Day {item.form.day}
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>
        </aside>

        <main className="lg:col-span-8">
          <div className="mb-5 rounded-2xl border border-slate-800 bg-gradient-to-br from-sky-500/10 to-emerald-400/10 p-6 shadow-soft">
            <div className="text-xs text-slate-400">Console</div>
            <div className="mt-1 text-2xl font-bold tracking-tight">New demand estimate</div>
            <div className="mt-2 text-sm text-slate-300">
              Create a fresh estimate, review the result, and save it automatically to your workspace.
            </div>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <section className="rounded-2xl border border-slate-800 bg-slate-950/60 p-5 shadow-soft">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">Inputs</h2>
                  <div className="text-xs text-slate-400">Complete all required fields</div>
                </div>
                <div className="hidden md:flex flex-wrap gap-2">
                  {metadata?.cvR2 ? <StatPill label="CV" value={metadata.cvR2.toFixed(3)} /> : null}
                </div>
              </div>

              {loadingMeta ? (
                <div className="space-y-3">
                  <Skeleton />
                  <Skeleton />
                  <Skeleton />
                </div>
              ) : metaError ? (
                <div className="rounded-xl border border-rose-900/60 bg-rose-950/40 p-4 text-sm text-rose-200">
                  <div className="font-semibold">Set up required</div>
                  <div className="mt-1 text-rose-200/80">Model assets are missing. Train once to unlock the console.</div>
                </div>
              ) : !form || !metadata ? null : (
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Field label="Geohash">
                    <input
                      value={form.geohash}
                      onChange={(e) => setForm({ ...form, geohash: e.target.value })}
                      list="geohashTop"
                      className={inputClass}
                      placeholder="e.g. qp02z1"
                    />
                    <datalist id="geohashTop">
                      {metadata.geohashTop.map((g) => (
                        <option key={g} value={g} />
                      ))}
                    </datalist>
                    <Help>Search suggestions or paste a geohash.</Help>
                  </Field>

                  <Field label="Day">
                    <select
                      value={form.day}
                      onChange={(e) => setForm({ ...form, day: Number(e.target.value) })}
                      className={inputClass}
                    >
                      {Array.from(
                        { length: metadata.dayRange.max - metadata.dayRange.min + 1 },
                        (_, i) => metadata.dayRange.min + i
                      ).map((d) => (
                        <option key={d} value={d}>
                          {d}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <Field label="Timestamp">
                    <select
                      value={form.timestamp}
                      onChange={(e) => setForm({ ...form, timestamp: e.target.value })}
                      className={inputClass}
                    >
                      {metadata.timestampOptions.slice(0, 400).map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <Field label="Road Type">
                    <select
                      value={form.RoadType}
                      onChange={(e) => setForm({ ...form, RoadType: e.target.value })}
                      className={inputClass}
                    >
                      {metadata.categorical.RoadType.map((x) => (
                        <option key={x} value={x}>
                          {x}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <Field label="Number of Lanes">
                    <select
                      value={form.NumberofLanes}
                      onChange={(e) => setForm({ ...form, NumberofLanes: Number(e.target.value) })}
                      className={inputClass}
                    >
                      {metadata.lanesOptions.map((x) => (
                        <option key={x} value={x}>
                          {x}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <Field label="Large Vehicles">
                    <select
                      value={form.LargeVehicles}
                      onChange={(e) => setForm({ ...form, LargeVehicles: e.target.value })}
                      className={inputClass}
                    >
                      {metadata.categorical.LargeVehicles.map((x) => (
                        <option key={x} value={x}>
                          {x}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <Field label="Landmarks">
                    <select
                      value={form.Landmarks}
                      onChange={(e) => setForm({ ...form, Landmarks: e.target.value })}
                      className={inputClass}
                    >
                      {metadata.categorical.Landmarks.map((x) => (
                        <option key={x} value={x}>
                          {x}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <Field label="Weather">
                    <select
                      value={form.Weather}
                      onChange={(e) => setForm({ ...form, Weather: e.target.value })}
                      className={inputClass}
                    >
                      {metadata.categorical.Weather.map((x) => (
                        <option key={x} value={x}>
                          {x}
                        </option>
                      ))}
                    </select>
                  </Field>

                  <Field label="Temperature">
                    <input
                      type="number"
                      step="0.1"
                      value={form.Temperature}
                      min={metadata.temperatureRange.min}
                      max={metadata.temperatureRange.max}
                      onChange={(e) => setForm({ ...form, Temperature: Number(e.target.value) })}
                      className={inputClass}
                    />
                    <Help>
                      Range: {metadata.temperatureRange.min.toFixed(1)} → {metadata.temperatureRange.max.toFixed(1)}
                    </Help>
                  </Field>

                  <div className="sm:col-span-2 pt-2">
                    <button
                      onClick={onPredict}
                      disabled={!canPredict || predicting}
                      className={classNames(
                        "w-full rounded-xl px-4 py-3 text-sm font-semibold transition",
                        predicting || !canPredict
                          ? "cursor-not-allowed bg-slate-800 text-slate-300"
                          : "bg-gradient-to-r from-sky-400 to-emerald-300 text-slate-950 hover:opacity-95"
                      )}
                    >
                      {predicting ? "Generating estimate…" : "Generate estimate"}
                    </button>
                    {!canPredict ? (
                      <div className="mt-2 text-xs text-slate-400">
                        Tip: geohash must be at least 5 characters (e.g. <span className="font-mono">qp02z</span>).
                      </div>
                    ) : null}
                  </div>
                </div>
              )}
            </section>

            <section className="rounded-2xl border border-slate-800 bg-slate-950/60 p-5 shadow-soft">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">Result</h2>
                  <div className="text-xs text-slate-400">Latest estimate</div>
                </div>
                <div className="text-xs text-slate-400">{BRAND}</div>
              </div>

              {predictError ? (
                <div className="rounded-xl border border-rose-900/60 bg-rose-950/40 p-4 text-sm text-rose-200">
                  <div className="font-semibold">Couldn’t generate estimate</div>
                  <div className="mt-1 text-rose-200/80">{predictError}</div>
                </div>
              ) : predicting ? (
                <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-6">
                  <div className="text-sm text-slate-300">Working…</div>
                  <div className="mt-3 space-y-3">
                    <Skeleton />
                    <Skeleton />
                  </div>
                </div>
              ) : prediction == null ? (
                <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-6">
                  <div className="text-sm text-slate-300">No estimate yet.</div>
                  <div className="mt-2 text-xs text-slate-400">Generate an estimate to see it here.</div>
                </div>
              ) : (
                <div className="rounded-2xl border border-emerald-500/30 bg-gradient-to-br from-emerald-500/15 to-sky-500/10 p-6">
                  <div className="text-xs uppercase tracking-wide text-slate-300">Demand</div>
                  <div className="mt-2 text-5xl font-bold tabular-nums">
                    {Number.isFinite(prediction) ? prediction.toFixed(6) : String(prediction)}
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <StatPill label="Workspace" value="Saved" />
                    <StatPill label="Latency" value="< 1s" />
                  </div>

                  <div className="mt-5 grid grid-cols-2 gap-2 text-xs text-slate-300">
                    <Mini label="Geohash" value={form?.geohash ?? "—"} />
                    <Mini label="Time" value={form ? `${form.timestamp} • Day ${form.day}` : "—"} />
                    <Mini label="Road" value={form?.RoadType ?? "—"} />
                    <Mini label="Weather" value={form?.Weather ?? "—"} />
                  </div>
                </div>
              )}

              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  onClick={() => {
                    setPrediction(null);
                    setPredictError(null);
                    setToast("Cleared result");
                  }}
                  className="rounded-xl border border-slate-800 bg-slate-950/40 px-3 py-2 text-xs font-semibold text-slate-200 hover:border-slate-700"
                  type="button"
                >
                  Clear result
                </button>
                <button
                  onClick={() => {
                    navigator.clipboard
                      .writeText(JSON.stringify({ inputs: form, demand: prediction }, null, 2))
                      .then(() => setToast("Copied result"))
                      .catch(() => setToast("Copy failed"));
                  }}
                  className="rounded-xl border border-slate-800 bg-slate-950/40 px-3 py-2 text-xs font-semibold text-slate-200 hover:border-slate-700"
                  type="button"
                >
                  Copy result
                </button>
              </div>
            </section>
          </div>

          <footer className="mt-6 flex flex-col gap-2 text-xs text-slate-500 md:flex-row md:items-center md:justify-between">
            <div>© {new Date().getFullYear()} {BRAND}</div>
            <div>Demand Console</div>
          </footer>
        </main>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: any }) {
  return (
    <label className="block">
      <div className="mb-1 text-xs font-semibold text-slate-200">{label}</div>
      {children}
    </label>
  );
}

function Help({ children }: { children: any }) {
  return <div className="mt-1 text-[11px] text-slate-400">{children}</div>;
}

function Skeleton() {
  return <div className="h-9 w-full animate-pulse rounded-xl bg-slate-800/60" />;
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-3">
      <div className="text-[11px] text-slate-400">{label}</div>
      <div className="mt-1 truncate font-semibold text-slate-200">{value}</div>
    </div>
  );
}

const inputClass =
  "w-full rounded-xl border border-slate-800 bg-slate-950/70 px-3 py-2.5 text-sm text-slate-100 outline-none ring-0 transition focus:border-slate-600";

