// Lets Node run the Worker sources directly, unbundled.
//
// Node 26 strips TypeScript natively but resolves relative imports by strict ESM
// rules, which demand a file extension. src/ is written for the Workers bundler
// and imports "./tz", not "./tz.ts". Rather than rewrite every import to suit a
// test harness -- or pull in a bundler purely to run one script -- this adds the
// extension during resolution.
//
// It matters that parity runs against the ACTUAL src/ files. A gate that tested
// a compiled or vendored copy would be proving something about the copy.
import { registerHooks } from "node:module";

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith(".") && !/\.\w+$/.test(specifier)) {
      try {
        return nextResolve(specifier + ".ts", context);
      } catch {
        // fall through: let Node report its own error for the original specifier
      }
    }
    return nextResolve(specifier, context);
  },
});
