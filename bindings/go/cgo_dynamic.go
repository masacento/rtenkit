//go:build !rtenkit_static

package rtenkit

/*
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/dist/rtenkit-darwin-arm64 -lrtenkit
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/dist/rtenkit-darwin-amd64 -lrtenkit
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/dist/rtenkit-linux-amd64 -lrtenkit　-lm -lpthread
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/dist/rtenkit-linux-arm64 -lrtenkit　-lm -lpthread
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/dist/rtenkit-windows-amd64 -lrtenkit
*/
import "C"
