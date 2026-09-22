import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* `standalone`: la build porta con sé soltanto i moduli che usa davvero, e
   * l'immagine di produzione non deve installare node_modules da capo. */
  output: "standalone",
};

export default nextConfig;
