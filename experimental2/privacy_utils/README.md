## Notes

This folder implements memory-efficient clipping procedure that clips by layer. The procedure is in fact not faster but
slower, due to large einsums that collapses large per-sample gradients for weights of linear layers.
