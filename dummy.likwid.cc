#include <likwid.h>

int
main()
{
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;

  LIKWID_MARKER_START("test");
  LIKWID_MARKER_STOP("test");

  LIKWID_MARKER_CLOSE;
}