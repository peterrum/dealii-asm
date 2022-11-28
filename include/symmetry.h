#pragma once

namespace SymmetryType
{
  enum SymmetryType
  {
    symmetric,
    non_symmetric,
    undefined
  };

  inline SymmetryType
  operator&(const SymmetryType f1, const SymmetryType f2)
  {
    if (f1 == symmetric && f2 == symmetric)
      return symmetric;

    if (f1 == undefined || f2 == undefined)
      return undefined;

    return non_symmetric;
  }

} // namespace SymmetryType
