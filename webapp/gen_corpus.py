#!/usr/bin/env python3
"""Generate a LARGE Northwind knowledge base so the working set exceeds GPU KV (the regime where
CacheBlend's blend-tier reuse actually fires and the $-saved scoreboard moves). Deterministic
(no RNG that breaks reproducibility): a product catalog with retrievable SKU + price facts, split
into per-category files under corpus/. ~500 products -> ~200 chunks @512 tok >> ~87-chunk GPU KV.

Run: python gen_corpus.py   (writes corpus/catalog-*.md)
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus")

CATEGORIES = {
    "networking": ["switch", "router", "access point", "SFP transceiver", "patch panel", "media converter",
                   "PoE injector", "network tap", "load balancer", "firewall appliance"],
    "cabling": ["Cat6 cable", "Cat6a cable", "fiber patch cable", "DAC cable", "console cable",
                "bulk cable spool", "keystone jack", "cable tester", "crimp tool", "fiber cassette"],
    "power": ["rack PDU", "UPS", "power inverter", "surge protector", "DC power supply",
              "battery pack", "transfer switch", "power meter", "generator interlock", "line conditioner"],
    "storage": ["NVMe SSD", "SATA SSD", "enterprise HDD", "NAS enclosure", "drive caddy",
                "RAID controller", "SAS expander", "tape drive", "M.2 adapter", "backup appliance"],
    "compute": ["1U server", "2U server", "GPU server", "edge node", "blade module",
                "server memory kit", "CPU tray", "cooling module", "management card", "riser card"],
}
ADJ = ["compact", "rugged", "high-density", "low-latency", "redundant", "field-serviceable",
       "hot-swappable", "energy-efficient", "enterprise-grade", "toolless"]
FEAT = ["a lockable front bezel", "dual redundant inputs", "front-to-back airflow", "a color status LCD",
        "quiet sub-30dB operation", "tool-free rail mounting", "an out-of-band management port",
        "surge and brownout protection", "hot-swappable modules", "a 5-year advance-replacement warranty"]

def price(i): return 49 + (i * 37) % 4200          # deterministic, retrievable
def sku(cat, i): return f"NW-{cat[:3].upper()}-{1000 + i}"

def entry(cat, i, kind):
    a1, a2 = ADJ[i % len(ADJ)], ADJ[(i * 3) % len(ADJ)]
    f1, f2 = FEAT[i % len(FEAT)], FEAT[(i * 5) % len(FEAT)]
    name = f"Northwind {kind.title()} {kind[:2].upper()}{200 + i}"
    return (
        f"### {name}\n"
        f"SKU: {sku(cat, i)}. Price: ${price(i)}. Category: {cat}.\n"
        f"The {name} is a {a1}, {a2} {kind} built for production {cat} deployments. It ships with "
        f"{f1} and {f2}. Rated for continuous duty in a standard 19-inch rack, it targets the "
        f"mid-market and branch-office segment where reliability and serviceability matter more than "
        f"peak specs. Typical lead time is {1 + i % 5} business days from the Reno warehouse, and it "
        f"carries the standard Northwind 2-year limited warranty. The {name} (SKU {sku(cat, i)}) lists "
        f"at ${price(i)} and is eligible for free ground shipping on orders over $75.\n"
    )

def main():
    os.makedirs(OUT, exist_ok=True)
    total = 0
    for cat, kinds in CATEGORIES.items():
        lines = [f"# Northwind Product Catalog — {cat.title()}\n"]
        i = 0
        for _ in range(20):                    # 20 passes x 10 kinds = 200 products / category
            for kind in kinds:
                lines.append(entry(cat, i, kind)); i += 1
        open(os.path.join(OUT, f"catalog-{cat}.md"), "w").write("\n".join(lines))
        total += i
        print(f"catalog-{cat}.md: {i} products")
    print(f"total products: {total}")

if __name__ == "__main__":
    main()
