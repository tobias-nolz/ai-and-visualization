/* Measure labels in the rendered plot so pixel fonts stay clear at any zoom. */
(() => {
    function arrange(graph) {
        const config = graph.layout && graph.layout.meta && graph.layout.meta.confusion_labels;
        graph.querySelectorAll('.confusion-label-leaders').forEach(el => el.remove());
        if (!config) {
            graph._confusionFigure = null;
            const canvas = document.getElementById('network-canvas');
            if (canvas) {
                canvas.style.minWidth = '700px';
                canvas.style.height = graph._fullLayout.height + 'px';
                graph.style.height = graph._fullLayout.height + 'px';
            }
            return; // Heatmap and empty-state annotations are unrelated.
        }
        const canvas = document.getElementById('network-canvas');
        if (canvas) {
            if (graph._confusionResizing) return;
            const metadata = graph.layout.meta;
            const figureKey = JSON.stringify(metadata);
            const newFigure = graph._confusionFigure !== figureKey;
            graph._confusionFigure = figureKey;
            canvas.style.minWidth = (metadata.minimum_width || 700) + 'px';
            canvas.style.height = (metadata.minimum_height || 640) + 'px';
            graph.style.height = canvas.style.height;
            const needsResize = Math.abs(graph._fullLayout.width - graph.clientWidth) > 1 || Math.abs(graph._fullLayout.height - graph.clientHeight) > 1;
            // A filter or matrix transition can retain expanded axis ranges.
            // Restore the new figure's ranges after sizing, while preserving
            // manual zoom and pan on subsequent draws of that same figure.
            if (needsResize || newFigure) {
                graph._confusionResizing = true;
                const sized = needsResize ? Plotly.Plots.resize(graph) : Promise.resolve();
                sized.then(() => {
                    const ranges = graph.layout.meta.initial_ranges;
                    if (ranges) return Plotly.relayout(graph, {
                        'xaxis.range':[ranges[0][0],ranges[1][0]],
                        'yaxis.range':[ranges[0][1],ranges[1][1]]
                    });
                }).finally(() => {
                    graph._confusionResizing = false;
                    requestAnimationFrame(() => arrange(graph));
                });
                return;
            }
        }
        const plot = graph._fullLayout;
        const origin = graph.getBoundingClientRect();
        const bounds = {left: origin.left + plot.xaxis._offset + 3,
            top: origin.top + plot.yaxis._offset + 3,
            right: origin.left + plot.xaxis._offset + plot.xaxis._length - 3,
            bottom: origin.top + plot.yaxis._offset + plot.yaxis._length - 3};
        const nodeTrace = graph._fullData.find(t => t.customdata && t.customdata.length && t.customdata[0][0] === 'node');
        if (!nodeTrace) return;
        const nodes = new Map();
        const markers = graph.querySelectorAll('.trace' + nodeTrace.uid + ' .point');
        markers.forEach((el, i) => {
            const b = el.getBoundingClientRect();
            nodes.set(Number(nodeTrace.customdata[i][1]), {x: (b.left+b.right)/2, y: (b.top+b.bottom)/2,
                r: Math.max(b.width, b.height)/2 + 3});
        });
        const labels = [];
        graph.querySelectorAll('.annotation').forEach(el => {
            const index = Number(el.getAttribute('data-index'));
            const annotation = graph.layout.annotations[index];
            if (!annotation || !/^species-/.test(annotation.name || '')) return;
            el.style.transform = '';
            el.style.visibility = '';
            const id = Number(annotation.name.slice(8));
            const node = nodes.get(id);
            const info = config.find(item => item.id === id);
            if (!node || !info) return;
            if (node.x < bounds.left || node.x > bounds.right || node.y < bounds.top || node.y > bounds.bottom) {
                el.style.visibility = 'hidden';
                return;
            }
            const box = el.getBoundingClientRect();
            const cx = origin.left + plot.xaxis._offset + plot.xaxis.l2p(info.center[0]);
            const cy = origin.top + plot.yaxis._offset + plot.yaxis.l2p(info.center[1]);
            labels.push({el, node, id, base: box, w: box.width, h: box.height,
                angle: Math.atan2(node.y-cy, node.x-cx), focused: annotation.font.color === '#1d4ed8', box: null});
        });
        // Sample the visible curves once; avoid placing text across an arrow.
        const edgePoints = [];
        graph.querySelectorAll('.scatterlayer path.js-line').forEach(path => {
            const matrix = path.getScreenCTM();
            if (!matrix) return;
            const length = path.getTotalLength();
            for (let d = 0; d <= length; d += 12) {
                const p = path.getPointAtLength(d);
                edgePoints.push(new DOMPoint(p.x, p.y).matrixTransform(matrix));
            }
        });
        const overlap = (a,b,pad=3) => Math.max(0, Math.min(a.right,b.right)+pad-Math.max(a.left,b.left)) *
            Math.max(0, Math.min(a.bottom,b.bottom)+pad-Math.max(a.top,b.top));
        function score(item, box, distance, direction) {
            const outside = Math.max(0,bounds.left-box.left) + Math.max(0,box.right-bounds.right) +
                Math.max(0,bounds.top-box.top) + Math.max(0,box.bottom-bounds.bottom);
            let cost = outside * 1e7;
            for (const node of nodes.values()) {
                const dx = Math.max(box.left-node.x, 0, node.x-box.right);
                const dy = Math.max(box.top-node.y, 0, node.y-box.bottom);
                cost += Math.max(0, node.r-Math.hypot(dx,dy)) * 1e6;
            }
            for (const other of labels) {
                if (other !== item && other.box) cost += overlap(box, other.box) * 1e4;
            }
            for (const p of edgePoints) {
                if (p.x > box.left-2 && p.x < box.right+2 && p.y > box.top-2 && p.y < box.bottom+2) cost += 45;
            }
            // Keep labels close, favoring the direction out of their component.
            const gap = Math.max(0, Math.hypot(
                Math.max(box.left-item.node.x,0,item.node.x-box.right),
                Math.max(box.top-item.node.y,0,item.node.y-box.bottom)) - item.node.r);
            return cost + gap * 4 + (1-Math.cos(direction-item.angle)) * 80;
        }
        function clear(item, box) {
            // Keep every visible label next to its own circle. If no nearby
            // slot is free, the full name is still available on node hover.
            const dx = Math.max(box.left-item.node.x,0,item.node.x-box.right);
            const dy = Math.max(box.top-item.node.y,0,item.node.y-box.bottom);
            if (Math.hypot(dx,dy) - item.node.r > 12) return false;
            if (box.left < bounds.left || box.right > bounds.right || box.top < bounds.top || box.bottom > bounds.bottom) return false;
            for (const node of nodes.values()) {
                const dx = Math.max(box.left-node.x, 0, node.x-box.right);
                const dy = Math.max(box.top-node.y, 0, node.y-box.bottom);
                if (Math.hypot(dx,dy) < node.r) return false;
            }
            return labels.every(other => other === item || !other.box || overlap(box,other.box) === 0);
        }
        // Keep focus and larger species nodes labeled first. ID breaks ties.
        labels.sort((a,b) => Number(b.focused)-Number(a.focused) || b.node.r-a.node.r || b.w-a.w || a.id-b.id);
        for (let pass=0; pass<3; pass++) {
            for (const item of labels) {
                // Retaining a valid previous placement makes refinement monotonic.
                let best = item.box && clear(item,item.box) ? {box:item.box,
                    cost:score(item,item.box,Math.hypot((item.box.left+item.box.right)/2-item.node.x,
                        (item.box.top+item.box.bottom)/2-item.node.y),item.angle)} : null;
                for (const extra of [0, 4, 8]) {
                    for (let i=0; i<16; i++) {
                        const angle = item.angle + i*Math.PI/8;
                        const dx = Math.cos(angle)*(item.node.r+5+item.w/2+extra);
                        const dy = Math.sin(angle)*(item.node.r+5+item.h/2+extra);
                        const x = item.node.x+dx, y = item.node.y+dy;
                        const box = {left:x-item.w/2, right:x+item.w/2, top:y-item.h/2, bottom:y+item.h/2};
                        if (!clear(item,box)) continue;
                        const cost = score(item, box, Math.hypot(dx,dy), angle);
                        if (!best || cost < best.cost) best = {box, cost};
                    }
                }
                item.box = best ? best.box : null;
            }
        }
        for (const item of labels) {
            if (!item.box) {
                item.el.style.visibility = 'hidden';
                continue;
            }
            const box = item.box;
            item.el.style.transform = `translate(${box.left-item.base.left}px,${box.top-item.base.top}px)`;
        }
        graph._confusionPlacedFigure = JSON.stringify(graph.layout.meta);
        graph.dataset.labelLayout = 'ready';
    }
    function connect() {
        const graph = document.querySelector('#network .js-plotly-plot');
        if (!graph || typeof graph.on !== 'function' || graph.dataset.labelPlacement) return;
        graph.dataset.labelPlacement = 'ready';
        let frame;
        const schedule = () => { graph.dataset.labelLayout = 'pending'; cancelAnimationFrame(frame); frame = requestAnimationFrame(() => arrange(graph)); };
        graph.on('plotly_beforeplot', () => { graph.dataset.labelLayout = 'pending'; });
        graph.on('plotly_afterplot', schedule);
        schedule();
        if (document.fonts) document.fonts.ready.then(schedule);
    }
    new MutationObserver(connect).observe(document.documentElement, {childList:true, subtree:true});
    connect();
})();