import { forwardRef } from "react";

function legacyClassName(variant, size, extra) {
  const classes = ["md-btn"];
  if (variant === "ghost") classes.push("md-btn-text");
  if (variant === "outline") classes.push("md-btn-outlined");
  if (variant === "secondary" || variant === "tonal") classes.push("md-btn-tonal");
  if (variant === "default") classes.push("md-btn-filled");
  if (size === "sm") classes.push("sm-btn");
  if (extra) classes.push(extra);
  return classes.join(" ");
}

function modernClassName(variant, size, extra) {
  const classes = ["btn"];
  if (variant === "ghost") classes.push("btn-ghost");
  if (variant === "outline") classes.push("btn-outline");
  if (variant === "secondary" || variant === "tonal") classes.push("btn-secondary");
  if (variant === "destructive") classes.push("btn-destructive");
  if (size === "sm") classes.push("btn-sm");
  if (size === "icon") classes.push("btn-icon");
  if (extra) classes.push(extra);
  return classes.join(" ");
}

const Button = forwardRef(function Button(
  {
    className = "",
    variant = "default",
    size = "default",
    type = "button",
    loading = false,
    children,
    legacy = true,
    ...props
  },
  ref
) {
  const mergedClassName = legacy ? legacyClassName(variant, size, className) : modernClassName(variant, size, className);

  return (
    <button
      ref={ref}
      type={type}
      className={mergedClassName}
      disabled={loading || props.disabled}
      aria-busy={loading ? "true" : undefined}
      {...props}
    >
      {loading ? <span className="btn-spinner" aria-hidden="true" /> : null}
      <span className={loading ? "btn-content is-loading" : "btn-content"}>{children}</span>
    </button>
  );
});

export { Button };
