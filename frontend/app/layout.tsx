import type { Metadata, Viewport } from "next";
import { Instrument_Serif, Literata, IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { Providers } from "./providers";
import "./globals.css";

/* Tre ruoli tipografici, tre facce distinte.
 *
 * Instrument Serif porta il carattere: stretta e verticale, dice «edizione»
 * senza il contrasto teatrale dei serif da titolo. Literata fa il lavoro vero,
 * cioè reggere paragrafi lunghi senza stancare. Plex Mono è la voce
 * dell'apparato: riferimenti, orari, identificativi — tutto ciò che si allinea
 * in colonna e si copia. */
const display = Instrument_Serif({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

const lettura = Literata({
  subsets: ["latin"],
  variable: "--font-lettura",
  display: "swap",
});

const apparato = IBM_Plex_Mono({
  weight: ["400", "500"],
  subsets: ["latin"],
  variable: "--font-apparato",
  display: "swap",
});

const interfaccia = IBM_Plex_Sans({
  weight: ["400", "500", "600"],
  subsets: ["latin"],
  variable: "--font-interfaccia",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Personalities",
  description: "Conversa con personalità ricostruite dai loro documenti.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  /* `viewportFit: cover` è ciò che rende disponibili le variabili
   * `env(safe-area-inset-*)`: senza, il composer finirebbe sotto la barra di
   * gesture su iPhone e i suoi ultimi millimetri sarebbero intoccabili. */
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#eceef1" },
    { media: "(prefers-color-scheme: dark)", color: "#121820" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="it">
      <body
        className={`${display.variable} ${lettura.variable} ${apparato.variable} ${interfaccia.variable}`}
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
