import React from 'react'
import { motion } from 'framer-motion'

const steps = [
  {
    num: "01",
    title: "Ingest Data",
    desc: "Fetch OHLCV data directly via yfinance API."
  },
  {
    num: "02",
    title: "Engineer Features",
    desc: "Calculate returns, volatility, and technical indicators."
  },
  {
    num: "03",
    title: "Detect Anomalies",
    desc: "Run Isolation Forest + Autoencoders for scoring."
  },
  {
    num: "04",
    title: "Ensemble Logic",
    desc: "Combine models. If both flag it, raise alert."
  },
  {
    num: "05",
    title: "Visualize Results",
    desc: "Render interactive charts in the React frontend."
  },
  {
    num: "06",
    title: "Forecast",
    desc: "Project volatility paths using LSTM/Transformer."
  }
]

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="py-24 px-6 bg-white w-full border-y border-card-border">
      <div className="max-w-7xl mx-auto">
        <div className="mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4 tracking-tight">
            How the engine works
          </h2>
          <p className="text-text-secondary text-lg max-w-2xl">
            A complete end-to-end pipeline from raw ticker data to actionable insights.
          </p>
        </div>

        <div className="relative">
          {/* Connecting Line (Desktop) */}
          <div className="hidden md:block absolute top-[28px] left-[40px] right-[40px] h-[2px] bg-card-border overflow-hidden">
            <motion.div 
              className="h-full bg-brand-blue"
              initial={{ width: "0%" }}
              whileInView={{ width: "100%" }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 1.5, ease: "easeInOut" }}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-6 gap-8 md:gap-4 relative z-10">
            {steps.map((step, i) => (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.15 }}
                key={i}
                className="flex flex-row md:flex-col items-start gap-4 md:gap-6"
              >
                <div className="w-14 h-14 shrink-0 rounded-full bg-page-bg border-2 border-brand-blue flex items-center justify-center font-mono font-bold text-brand-blue text-lg shadow-lg relative group-hover:scale-110 transition-transform">
                   {step.num}
                   {/* Mobile connecting line */}
                   {i !== steps.length - 1 && (
                     <div className="block md:hidden absolute top-[56px] left-1/2 -ml-[1px] w-[2px] h-[32px] bg-card-border" />
                   )}
                </div>
                <div>
                  <h4 className="font-bold text-text-primary mb-1 md:text-sm lg:text-base">{step.title}</h4>
                  <p className="text-text-secondary text-sm leading-snug">{step.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
