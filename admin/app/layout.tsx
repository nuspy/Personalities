import type { Metadata, Viewport } from "next";
import { Instrument_Serif, IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { Providers } from "./providers";
import { Guscio } from "./guscio";
import "./globals.css";

/* Due facce e non tre: la console non ha prosa da leggere.
 *
 * Instrument Serif resta per l'identità — è la stessa piattaforma — ma il
 * corpo passa a Plex Sans, perché qui si leggono etichette, nomi e numeri, e
 * un serif da lettura lunga su una tabella stanca senza dare nulla in cambio.
 * Plex Mono porta gli identificativi e i punteggi, che vanno allineati in
 * colonna e copiati. */
const display = Instrument_Serif({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const interfaccia = IBM_Plex_Sans({
  weight: ["400", "500", "600"],
  subsets: ["latin"],
  variable: "--font-interfaccia",
  display: "swap",
});

const apparato = IBM_Plex_Mono({
  weight: ["400", "500"],
  subsets: ["latin"],
  variable: "--font-apparato",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Console — Personalities",
  description: "Amministrazione delle personalità e dei corpora.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="it">
      <body
        className={`${display.variable} ${interfaccia.variable} ${apparato.variable}`}
        /* La console si legge in sans: il serif da lettura qui non serve. */
        style={{ fontFamily: "var(--font-interfaccia), system-ui, sans-serif" }}
      >
        <Providers>
          <Guscio>{children}</Guscio>
        </Providers>
      </body>
    </html>
  );
}
