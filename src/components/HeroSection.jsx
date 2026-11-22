import React from 'react';
import { ArrowRight } from 'lucide-react';

const TIERS = [
  { name: 'Benign', body: 'Looks harmless.', dot: 'bg-derma-teal' },
  { name: 'Atypical', body: 'Has unusual features and is worth monitoring.', dot: 'bg-derma-yellow' },
  { name: 'Pre-malignant', body: 'Could turn into cancer if left alone.', dot: 'bg-derma-amber' },
  { name: 'Malignant', body: 'Has features of skin cancer.', dot: 'bg-derma-coral' },
];

export default function HeroSection({ onStartTriage }) {
  return (
    <section className="border-b border-derma-border">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16 lg:py-20">
        <div className="grid lg:grid-cols-12 gap-10 lg:gap-14 items-start">
          <div className="lg:col-span-7">
            <p className="text-sm text-derma-accent font-semibold">
              Decision support for clinicians
            </p>
            <h1 className="mt-3 text-4xl sm:text-5xl lg:text-[3.5rem] font-semibold text-slate-50 leading-[1.1]">
              A clearer first look at a suspicious mole.
            </h1>
            <p className="mt-5 text-lg text-slate-300 max-w-xl leading-relaxed">
              SkinScan reads a lesion image together with a few patient details and sorts it into
              one of four risk tiers. It also shows you where on the lesion it was looking, so you
              can judge whether that makes sense.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-x-5 gap-y-3">
              <button
                onClick={onStartTriage}
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-derma-accent hover:bg-derma-accentHover text-slate-950 font-semibold transition-colors"
              >
                Start a scan
                <ArrowRight className="w-4 h-4" />
              </button>
              <p className="text-sm text-derma-muted max-w-xs">
                It flags lesions for a closer look. It doesn&apos;t diagnose, and a clinician should
                make the final call.
              </p>
            </div>
          </div>

          <div className="lg:col-span-5 rounded-xl border border-derma-border bg-derma-card p-6">
            <h2 className="text-xl font-semibold text-slate-50">The four risk tiers</h2>
            <p className="mt-1 text-sm text-derma-muted">What each result means, in plain words.</p>
            <ul className="mt-5 divide-y divide-derma-border">
              {TIERS.map((tier) => (
                <li key={tier.name} className="flex items-start gap-3 py-3 first:pt-0 last:pb-0">
                  <span className={`mt-2 w-2.5 h-2.5 rounded-full flex-shrink-0 ${tier.dot}`} />
                  <div>
                    <p className="font-semibold text-slate-50">{tier.name}</p>
                    <p className="text-sm text-slate-300">{tier.body}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <p className="mt-12 text-sm text-derma-muted">
          Built as a B.M.S. College of Engineering CSE major project (phase 2). Trained on HAM10000.
        </p>
      </div>
    </section>
  );
}
