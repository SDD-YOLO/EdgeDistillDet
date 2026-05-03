import { useEffect, useRef, useState } from "react";

export function M3Select({
  value,
  onChange,
  options,
  disabled,
  className = "",
  ariaLabel = "选择项",
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const normalizedValue = value == null ? "" : String(value);
  const normalizedOptions = Array.isArray(options)
    ? options.map((opt) => ({ ...opt, value: String(opt.value) }))
    : [];
  const selected =
    normalizedValue === ""
      ? undefined
      : normalizedOptions.find((opt) => opt.value === normalizedValue) ||
        normalizedOptions[0];

  useEffect(() => {
    if (!open) return undefined;
    const onDocPointerDown = (event) => {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(event.target)) {
        setOpen(false);
      }
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onDocPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onDocPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div
      ref={rootRef}
      className={`m3-select ${open ? "open" : ""} ${
        disabled ? "disabled" : ""
      } ${className}`.trim()}
    >
      <button
        type="button"
        className="m3-select-trigger md-input"
        onClick={() => {
          if (disabled) return;
          setOpen((prev) => !prev);
        }}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={ariaLabel}
        disabled={disabled}
      >
        <span className="m3-select-value">{selected?.label || ""}</span>
        <span className="material-icons m3-select-arrow">expand_more</span>
      </button>
      {open ? (
        <div className="m3-select-menu" role="listbox">
          {normalizedOptions.map((opt) => (
            <button
              key={opt.value}
              type="button"
              role="option"
              aria-selected={opt.value === normalizedValue}
              className={`m3-select-option ${
                opt.value === normalizedValue ? "selected" : ""
              }`}
              onClick={() => {
                onChange(opt.value);
                setOpen(false);
              }}
            >
              <span>{opt.label}</span>
              {opt.value === normalizedValue ? (
                <span className="material-icons">check</span>
              ) : null}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
