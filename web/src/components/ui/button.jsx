import { forwardRef } from "react";

const VARIANT_CLASS = {
  default: "md-btn md-btn-filled",
  secondary: "md-btn md-btn-tonal",
  outline: "md-btn md-btn-outlined",
  ghost: "md-btn md-btn-text",
  destructive: "md-btn"
};

const SIZE_CLASS = {
  default: "",
  sm: "sm-btn",
  icon: ""
};

function mergeClassNames(...parts) {
  return parts.filter(Boolean).join(" ");
}

const Button = forwardRef(function Button(
  { className = "", variant = "default", size = "default", type = "button", ...props },
  ref
) {
  return (
    <button
      ref={ref}
      type={type}
      className={mergeClassNames("md-btn", VARIANT_CLASS[variant], SIZE_CLASS[size], className)}
      {...props}
    />
  );
});

export { Button };
