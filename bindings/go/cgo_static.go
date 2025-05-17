//go:build rtenkit_static

package rtenkit

/*
#cgo darwin,arm64 LDFLAGS: ${SRCDIR}/dist/rtenkit-darwin-arm64/librtenkit.a
#cgo darwin,amd64 LDFLAGS: ${SRCDIR}/dist/rtenkit-darwin-amd64/librtenkit.a
#cgo linux,amd64 LDFLAGS: ${SRCDIR}/dist/rtenkit-linux-amd64/librtenkit.a　-lm -lpthread
#cgo linux,arm64 LDFLAGS: ${SRCDIR}/dist/rtenkit-linux-arm64/librtenkit.a　-lm -lpthread
#cgo windows,amd64 LDFLAGS: ${SRCDIR}/dist/rtenkit-windows-amd64/librtenkit.a
*/
import "C"
