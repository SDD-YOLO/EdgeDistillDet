import { forwardRef } from "react";

function mergeClassNames(...parts) {
  return parts.filter(Boolean).join(" ");
}

const Input = forwardRef(function Input({ className = "", ...props }, ref) {
  return (
    <input
      ref={ref}
      className={mergeClassNames("md-input", className)}
      {...props}
    />
  );
});

export { Input };
